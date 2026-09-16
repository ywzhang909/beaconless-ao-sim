import torch
import torch.optim as optim
from .fcnn import FCNN
from .util import um, mm, cm
import numpy as np
import matplotlib.pyplot as plt

"""
Copyright (c) 2026, Rafael de la Fuente
All rights reserved.
"""



# ----------------------------------------------------------------------
#  PINN solver
# ----------------------------------------------------------------------
class Farfield_PINN:
    """Physics‑informed NN for designing the far-field response **∂P/(∂Ωcosθ)** 
    of flat-optics devices.
    Given a desired far-field angular distribution and a intensity profile 
    at the *source* plane **I(x, y)**, the class trains a fully connected neural network 
    Phi(x, y) that parametrises an unknown phase profile.

    Parameters
    ----------
    Irad_fun : Callable[[Tensor, Tensor], Tensor]
        Target radiant intensity per cosine **∂P/(∂Ωcosθ)** as a function of direction
        cosines (α,β).  These variables are dimensionless and already live in
        Fourier space so no additional scaling is applied when calling this
        helper.
    I_fun : Callable[[Tensor, Tensor], Tensor]
        Source‑plane irradiance **I(x, y)** in physical units (metres).  Must
        accept tensors broadcast‑compatible with *(x,y)*.
    extent_source : float
        Side length of the *square* source aperture in the same units used for
        *x* and *y*.  Internally we store **M=extent_source∕2** for
        convenience.
    λ : float
        Wavelength.
    phi_scale : float, default ``8``
        Constant dividing the raw network output before it enters the PDE.
        Lower values make the initial Φ flatter.
    integration_points : int, default ``5000``
        Monte‑Carlo points for power normalisation between *I* and *I_rad*.
    hidden_layers, num_features : int
        Architecture of the underlying :class:`~fcnn.FCNN`.
    constrain : bool, default ``False``
        Forwarded to *FCNN* to optionally add a hard analytical term to Φ.
    symmetry : {``None``, ``"x"``, ``"y"``, ``"xy"``}, default ``"xy"``
        Reduces the solved domain by a factor of 2 or 4 when physical symmetry
        is present (e.g. circular aperture).
        Geometric symmetry reduction; chooses a quarter‑, half‑ or full domain.
    constrain_fn : Callable[[Tensor], Tensor] | None, optional
        Analytical function *g(x)* to include in the hard constraint path.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        I_fun,
        Irad_fun,
        extent_source,
        λ, 
        phi_scale = 8,
        integration_points: int = 5000,
        hidden_layers: int     = 3,
        num_features: int      = 150,
        constrain: bool   = False,
        symmetry: str | None   = "xy",
        constrain_fn=None,
    ):
        # ---------- domain selection ----------------------------------
        sym = None if symmetry is None else symmetry.lower()
        if   sym == "xy":         # quarter domain
            self.x_range, self.y_range = (0, 1), (0, 1)
        elif sym == "x":          # half domain: x‑symmetry only
            self.x_range, self.y_range = (0, 1), (-1, 1)
        elif sym == "y":          # half domain: y‑symmetry only
            self.x_range, self.y_range = (-1, 1), (0, 1)
        elif sym is None:         # full domain
            self.x_range, self.y_range = (-1, 1), (-1, 1)
        else:
            raise ValueError("symmetry must be 'xy', 'x', 'y' or None")
        self.symmetry = sym

        # -------------- rest of constructor -------------------------------
        # (normalise Irad_fun, build network, etc.)
        # ------------------------------------------------------------------
        # ---------------- optical constants & scaling factors ---------
        self.M       = extent_source / 2
        self.phi_scale = phi_scale    # scaling hyperparameter
        self.λ = λ
        
        N       = integration_points
        dx = dy = 2.0 / N
        x_bar   = dx * (torch.arange(N) - N // 2)
        y_bar   = dy * (torch.arange(N) - N // 2)
        xx_bar, yy_bar = torch.meshgrid(x_bar, y_bar, indexing='xy')
        α, β = x_bar, y_bar
        αα, ββ = xx_bar, yy_bar

        I_vals  = I_fun(self.M * xx_bar, self.M * yy_bar)
        Irad_vals  = Irad_fun(αα, ββ)
        area_I     = torch.trapz(torch.trapz(I_vals, x=self.M*x_bar, dim=1), x=self.M*y_bar, dim=0)
        area_Irad    = torch.trapz(torch.trapz(Irad_vals, x=α, dim=1), x=β, dim=0)

        
        self.area_ratio = area_I / area_Irad # ensures power conservation

        self.I_fun = lambda a, b: I_fun(a, b)
        self.Irad_fun_norm = lambda a, b: self.area_ratio * Irad_fun(a, b)

        # ---------- neural phase field --------------------------------
        self.net = FCNN(2, hidden_layers, num_features, 1,
                        constrain, constrain_fn)
        self.net.train()

    # ==================================================================
    #  Sampling utilities                                             #
    # ==================================================================
    def _grid(self, n, rng):
        """Return *n* equally‑spaced samples between *rng[0]* and *rng[1]*."""
        return torch.linspace(rng[0], rng[1], n)

    def sample_interior_points(self, N, noise_amplitude=0.001):
        """Quasi‑uniform sampling of collocation points inside the domain."""
        m  = int(np.ceil(np.sqrt(N)))
        X, Y = torch.meshgrid(self._grid(m, self.x_range),
                              self._grid(m, self.y_range),
                              indexing="xy")
        pts  = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)[:N]
        return pts + noise_amplitude * torch.randn_like(pts) if noise_amplitude else pts

    # -------- x‑edge --------------------------------------------------
    def sample_boundary_x(self, N, noise_amplitude=0.001):
        """Return *N* samples on the left x‑edge if x‑symmetry is active."""
        if self.symmetry not in ("x", "xy"):
            return torch.empty(0, 2)
        y  = self._grid(N, self.y_range).unsqueeze(1)
        pts = torch.cat([torch.full_like(y, self.x_range[0]), y], 1)
        return pts + noise_amplitude * torch.randn_like(pts) if noise_amplitude else pts

    # -------- y‑edge --------------------------------------------------
    def sample_boundary_y(self, N, noise_amplitude=0.001):
        """Return *N* samples on the bottom y‑edge if y‑symmetry is active."""
        if self.symmetry not in ("y", "xy"):
            return torch.empty(0, 2)
        x  = self._grid(N, self.x_range).unsqueeze(1)
        pts = torch.cat([x, torch.full_like(x, self.y_range[0])], 1)
        return pts + noise_amplitude * torch.randn_like(pts) if noise_amplitude else pts

    def boundary_loss_x(self, xb):
        """Enforce Neumann BC (∂Φ/∂x=0) on the x‑symmetry plane."""
        if self.symmetry not in ("x", "xy") or xb.numel() == 0:
            return torch.zeros(1, device=xb.device)
        xb.requires_grad_(True)
        grad = torch.autograd.grad(self.net(xb), xb,
                                   torch.ones_like(xb[:, :1]),
                                   create_graph=True)[0]
        return grad[:, 0:1]

    def boundary_loss_y(self, xb):
        """Enforce Neumann BC (∂Φ/∂y=0) on the y‑symmetry plane."""
        if self.symmetry not in ("y", "xy") or xb.numel() == 0:
            return torch.zeros(1, device=xb.device)
        xb.requires_grad_(True)
        grad = torch.autograd.grad(self.net(xb), xb,
                                   torch.ones_like(xb[:, :1]),
                                   create_graph=True)[0]
        return grad[:, 1:2]

    # ==================================================================
    # Loss functions: PDE residual and boundary losses
    # ==================================================================
    def pde_residual(self, x, residual_power):
        """

        Point‑wise residual for the far‑field ray‑mapping Monge–Ampère PDE. 
        Return |I_rad(α,β)·det(HessΦ)−I(x,y)|^p for each collocation point.

        Full PDE:

            I_rad(Φ_x, Φ_y) · (Φ_xx · Φ_yy − Φ_xy²)  =  I(x, y)

        where

            Φ_x  = ∂Φ/∂x,   Φ_y  = ∂Φ/∂y
            Φ_xx = ∂²Φ/∂x², Φ_yy = ∂²Φ/∂y², Φ_xy = ∂²Φ/∂x∂y
            I_rad is evaluated at (α, β) = (Φ_x, Φ_y)
            I      is evaluated at physical coords (M x, M y)

        The residual is the absolute difference raised to the exponent
        ``residual_power`` (1 for an L¹ residual, 2 for L², etc.). 


        Parameters
        ----------
        x : torch.Tensor, shape (N,2)
            Collocation points in the normalised square (−1,1)².
        power : int
            Exponent *p* applied to the absolute residual (use 1 for L¹ or 2 for L²).

        Returns
        -------
        torch.Tensor, shape (N,1)
            Point‑wise residual values.
        """
        x.requires_grad_(True)
        phi = self.net(x)  / self.phi_scale
        grad_phi = torch.autograd.grad(phi, x,
                                       grad_outputs=torch.ones_like(phi),
                                       create_graph=True)[0]
        dphi_dx = grad_phi[:, 0:1]
        dphi_dy = grad_phi[:, 1:2]
        d2phi_dx2 = torch.autograd.grad(dphi_dx, x,
                                        grad_outputs=torch.ones_like(dphi_dx),
                                        create_graph=True)[0][:, 0:1]
        d2phi_dy2 = torch.autograd.grad(dphi_dy, x,
                                        grad_outputs=torch.ones_like(dphi_dy),
                                        create_graph=True)[0][:, 1:2]
        d2phi_dxdy = torch.autograd.grad(dphi_dx, x,
                                         grad_outputs=torch.ones_like(dphi_dx),
                                         create_graph=True)[0][:, 1:2]
        alpha = dphi_dx
        beta  = dphi_dy
        L_val = self.Irad_fun_norm(alpha, beta)/ (self.M**2)
        lhs = L_val * torch.abs(d2phi_dx2 * d2phi_dy2 - d2phi_dxdy**2)
        rhs = self.I_fun(self.M * x[:, 0:1], self.M * x[:, 1:2])
        return torch.abs(lhs - rhs)**residual_power

    # ==================================================================
    # Compute test loss on a fixed grid
    # ==================================================================
    def compute_test_loss(self, num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise):
        x_test = self.sample_interior_points(num_test_interior, effective_noise)
        boundary_x_test = self.sample_boundary_x(num_test_boundary, effective_noise)
        boundary_y_test = self.sample_boundary_y(num_test_boundary, effective_noise)
        interior_loss = torch.mean(self.pde_residual(x_test, residual_power)**2)
        if boundary_x_test.numel() > 0:
            bc_loss_x = torch.mean(self.boundary_loss_x(boundary_x_test)**2)
        else:
            bc_loss_x = 0.0
        bc_loss_y = torch.mean(self.boundary_loss_y(boundary_y_test)**2)
        total_loss = (loss_weights[0] * interior_loss +
                      loss_weights[1] * bc_loss_x +
                      loss_weights[2] * bc_loss_y)
        return total_loss.item(), interior_loss.item(), bc_loss_x, bc_loss_y

    # ==================================================================
    #  Adam phase
    # ==================================================================
    def run_phase_adam(self, iterations, num_domain, num_boundary,
                       residual_power, loss_weights, effective_noise,
                       num_test_interior, num_test_boundary, lr):
        """Stochastic Adam training loop.

        Parameters
        ----------
        iterations : int
            Number of optimiser steps.
        num_domain : int
            Interior sample count per iteration.
        num_boundary : int
            Boundary sample count per iteration (per axis).
        residual_power : float
            Exponent *p* in the PDE residual ``|lhs − rhs|**p``; 2 is common.
        loss_weights : Sequence[float]
            Triplet ``(w_int, w_bx, w_by)`` scaling interior and boundary terms.
        effective_noise : float
            Standard deviation of Gaussian jitter added to every sampled point
            (helps avoid aliasing when using deterministic grids).
        num_test_interior, num_test_boundary : int
            Size of *held-out* test sets printed every 100 iterations.
        lr : float
            Adam learning rate.

        Notes
        -----
        *   Progress is sent to *stdout* every 100 steps – adapt if you redirect
            logs to a file.
        *   Gradients are zeroed **before** sampling so that memory usage stays
            constant regardless of batch size.
        """

        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for it in range(iterations):
            optimizer.zero_grad()
            x_domain = self.sample_interior_points(num_domain, effective_noise)
            boundary_x = self.sample_boundary_x(num_boundary, effective_noise)
            boundary_y = self.sample_boundary_y(num_boundary, effective_noise)
            interior_loss = torch.mean(self.pde_residual(x_domain, residual_power)**2)
            bc_loss_x = torch.mean(self.boundary_loss_x(boundary_x)**2) if boundary_x.numel() > 0 else 0.0
            bc_loss_y = torch.mean(self.boundary_loss_y(boundary_y)**2)
            total_loss = (loss_weights[0] * interior_loss +
                          loss_weights[1] * bc_loss_x +
                          loss_weights[2] * bc_loss_y)
            total_loss.backward()
            optimizer.step()
            if it % 100 == 0:
                test_total, test_interior, test_bc_x, test_bc_y = self.compute_test_loss(
                    num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise)
                # If bc_loss_x or bc_loss_y are floats, they don't have .item(); so we check:
                bc_x_val = bc_loss_x if isinstance(bc_loss_x, float) else bc_loss_x.item()
                bc_y_val = bc_loss_y if isinstance(bc_loss_y, float) else bc_loss_y.item()
                print(f"Adam Iter {it:5d}: Loss = {total_loss.item():.8f} "
                      f"(Interior = {interior_loss.item():.8f}, BC_x = {bc_x_val:.8f}, BC_y = {bc_y_val:.8f})")
                print(f"           Test Loss = {test_total:.8f} "
                      f"(Interior = {test_interior:.8f}, BC_x = {test_bc_x:.8f}, BC_y = {test_bc_y:.8f})")
    
    # ==================================================================
    #  LBFGS phase
    # ==================================================================
    def run_phase_lbfgs(self, num_domain, num_boundary, residual_power, loss_weights,
                        effective_noise, num_test_interior, num_test_boundary,
                        max_iterations_lbfgs, lr_lbfgs):
        """Batch L‑BFGS optimisation over a *static* sample set.

        Unlike the Adam stage, all domain and boundary points are drawn **once**
        before the first quasi‑Newton step and then reused.  This dramatically
        reduces variance in the line‑search but also risks over‑fitting if the
        sample counts are too low.

        Parameters
        ----------
        num_domain, num_boundary : int
            Training set sizes for interior and boundary respectively.
        residual_power : float
            Exponent *p* in the PDE residual.
        loss_weights : Sequence[float]
            Triplet ``(w_int, w_bx, w_by)``.
        effective_noise : float
            Jitter applied **once** during the initial sampling.
        num_test_interior, num_test_boundary : int
            Point counts for periodic test‑loss evaluation (every 100 calls to
            the closure).
        max_iterations_lbfgs : int
            Upper bound on the number of quasi‑Newton steps.
        lr_lbfgs : float
            Initial step size for the strong‑Wolfe line search.
        """

        x_domain = self.sample_interior_points(num_domain, effective_noise)
        boundary_x = self.sample_boundary_x(num_boundary, effective_noise)
        boundary_y = self.sample_boundary_y(num_boundary, effective_noise)
        lbfgs_iter = [0]
        def closure():
            optimizer_lbfgs.zero_grad()
            interior_loss = torch.mean(self.pde_residual(x_domain, residual_power)**2)
            bc_loss_x = torch.mean(self.boundary_loss_x(boundary_x)**2) if boundary_x.numel() > 0 else 0.0
            bc_loss_y = torch.mean(self.boundary_loss_y(boundary_y)**2)
            loss = (loss_weights[0] * interior_loss +
                    loss_weights[1] * bc_loss_x +
                    loss_weights[2] * bc_loss_y)
            loss.backward()
            lbfgs_iter[0] += 1
            if lbfgs_iter[0] % 100 == 0:
                test_total, test_interior, test_bc_x, test_bc_y = self.compute_test_loss(
                    num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise)
                bc_x_val = test_bc_x if isinstance(test_bc_x, float) else test_bc_x.item()
                bc_y_val = test_bc_y if isinstance(test_bc_y, float) else test_bc_y.item()
                print(f"LBFGS Iter {lbfgs_iter[0]:5d}: Loss = {loss.item():.8f} "
                      f"Test Loss = {test_total:.8f} (Interior = {test_interior:.8f}, BC_x = {bc_x_val:.8f}, BC_y = {bc_y_val:.8f})")
            return loss
        optimizer_lbfgs = optim.LBFGS(self.net.parameters(), max_iter=max_iterations_lbfgs,
                                      lr=lr_lbfgs, line_search_fn="strong_wolfe")
        optimizer_lbfgs.step(closure)

    # -------------------------
    # Main phase method.
    # The user calls this method each time to perform a single training phase.
    # Specify optimizer_type as 'adam', 'lbfgs', or 'deepxde'.
    # -------------------------
    def run_phase(self, optimizer_type, **kwargs):
        if optimizer_type.lower() == 'adam':
            self.run_phase_adam(**kwargs)
        elif optimizer_type.lower() == 'lbfgs':
            self.run_phase_lbfgs(**kwargs)
        elif optimizer_type.lower() == 'deepxde':
            self.residual_power_deepxde = kwargs.get("residual_power", 1.0)
            self.run_phase_deepxde(**kwargs)
        else:
            raise ValueError("Unsupported optimizer type. Use 'adam', 'lbfgs', or 'deepxde'.")

    # -------------------------
    # Utility method to print final test losses.
    # -------------------------
    def print_final_test_loss(self, num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise):
        final_total, final_interior, final_bc_x, final_bc_y = self.compute_test_loss(
            num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise)
        print("\nFinal Test Losses:")
        print(f"Total = {final_total:.8f}, Interior = {final_interior:.8f}, BC_x = {final_bc_x:.8f}, BC_y = {final_bc_y:.8f}")


    def evaluate_phi(self, x, y):

        """Evaluate the predicted phase **Phi(x, y)** on arbitrary input grids.

        The routine mirrors the coordinates back into the *learned* domain when
        symmetry reduction (``'x'``, ``'y'``, or ``'xy'``) is active, converts
        them into the *normalised* network input space ``(X/M, Y/M)``, and
        finally rescales the raw network output to **physical radians** using

            Phi(x, y) = FCNN(x, y) * k  * M / phi_scale  

        Parameters
        ----------
        x, y : torch.Tensor
            Matching-shaped tensors of physical coordinates in the *source*
            plane (same units as ``extent_source``).

        Returns
        -------
        torch.Tensor
            *Flattened* tensor of shape ``(x.numel(), 1)`` containing the phase
            in radians.  Use ``reshape_like(x)`` to obtain the original grid
            shape if needed.

        Notes
        -----
        * The method does **not** perform any device transfers – ensure ``x``
          and ``y`` live on the same device as ``self.net``.
        """
        if   self.symmetry == 'xy':
            x, y = torch.abs(x), torch.abs(y)
        elif self.symmetry == 'x':
            x, y = torch.abs(x), y
        elif self.symmetry == 'y':
            x, y = x, torch.abs(y)

        k = 2.0 * np.pi / self.λ
        XY = torch.cat([x.ravel().reshape(-1,1), y.ravel().reshape(-1,1)], dim=1)
        with torch.no_grad():
            phi_pred = self.net((XY/self.M)) * k  * self.M    /(self.phi_scale)  
            
        return phi_pred


    def get_phi_spline(self, shape = (1000,1000)):
        """Return a bicubic *RectBivariateSpline* interpolant for **Phi(x, y)**.

        A dense Cartesian grid of size ``shape = (ny, nx)`` is sampled across
        the square ``[-M, M]²`` in the *source* plane, the phase is evaluated
        with evaluate_phi method, and the result fed into a scipy.interpolate.RectBivariateSpline

        Parameters
        ----------
        shape : tuple[int, int], default ``(1000, 1000)``
            Number of samples along the **y‑** and **x‑axis** respectively.

        Returns
        -------
        scipy.interpolate.RectBivariateSpline
            A bicubic spline that can be queried at arbitrary physical
            coordinates within the aperture.

        """
        
        ny,nx  = shape
        x = torch.linspace(-self.M, self.M, nx)
        y = torch.linspace(-self.M, self.M, ny)
        X, Y = torch.meshgrid(x, y, indexing='xy')

        phi_pred = self.evaluate_phi(X,Y).reshape(ny, nx)
                    
        # --- move to CPU for SciPy ------------------------------------
        if phi_pred.device.type != 'cpu':
            phi_pred = phi_pred.cpu().numpy()
            x = x.cpu().numpy()
            y = y.cpu().numpy()

        else:
            phi_pred = phi_pred.numpy()
            x = x.numpy()
            y = y.numpy()

        from scipy.interpolate import RectBivariateSpline
        
        fun_phi = RectBivariateSpline(x, y, phi_pred)
        return fun_phi






    def _configure_axes_unit_labels(self, ax, units):  # noqa: D401
        """Internal helper – set *x/y* labels according to *units* constant."""
        label = (
            "mm" if units is mm else "µm" if units is um else "cm" if units is cm else "nm"
        )
        ax.set_xlabel(f"x [{label}]")
        ax.set_ylabel(f"y [{label}]")



    def plot_source_irradiance(self, figsize=(5, 4),  units = mm, grid = False, res = (2048,2048)):
        """Render the *source*‑plane irradiance $I(x, y)$.

        A dense Cartesian grid is sampled across $[-M, M]^2$ and passed through
        the analytical intensity function ``I_fun``.  The result is displayed
        with *matplotlib*.

        Parameters
        ----------
        figsize : tuple[int, int], default ``(5,4)``
            Figure size in inches.
        units : {``mm``, ``um``, ``cm``, ``nm``}, default ``mm``
            Conversion factor translating physical metres to axis units.
        grid : bool, default ``False``
            Draw a faint grid behind the image to aid reading.
        res : tuple[int, int], default ``(2048,2048)``
            Sampling resolution ``(nx,ny)`` of the generated image.

        Notes
        -----
        * The method calls ``plt.show()`` and therefore blocks in interactive
          back‑ends.
        * Aspect ratio is forced to 1 to preserve square pixels even when the
          figure window is resized.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if grid == True:
            ax.grid(alpha =0.2)

        self._configure_axes_unit_labels(ax, units)

        nx, ny = res
        x = torch.linspace(-self.M, self.M, nx)
        y = torch.linspace(-self.M, self.M, ny)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        im = ax.imshow(self.I_fun(X, Y).cpu().numpy(), origin = 'lower', cmap ='inferno', extent = [float(x[0])/units, float(x[-1])/units, float(y[0])/units,  float(y[-1])/units ], aspect = 'auto')
        cb = fig.colorbar(im, orientation = 'vertical',fraction=0.045, label = 'Intensity [a.u.]')
        ax.set_title(r"Source plane irradiance $I(x,y)$")
        ax.set_aspect(1) 
        plt.show()





    def plot_target_radiant_intensity(self, figsize=(5, 4),  grid = False, res = (2048,2048)):
        """Render the *target* radiant intensity per cosine **∂P/(∂Ωcosθ)** (power‑normalised).

        Identical interface to :py:meth:`plot_source_irradiance`; only the
        plotted function differs (``Irad_fun_norm`` instead of ``I_fun``) and the
        spatial range is ``[-1, 1]``.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if grid == True:
            ax.grid(alpha =0.2)

        ax.set_xlabel(f"α")
        ax.set_ylabel(f"β")

        nα, nβ = res
        α = torch.linspace(-1.0, 1.0, nα)
        β = torch.linspace(-1.0, 1.0, nβ)
        αα, ββ = torch.meshgrid(α, β, indexing='xy')
        im = ax.imshow(self.Irad_fun_norm(αα, ββ).cpu().numpy(), origin = 'lower', cmap ='inferno', extent = [float(α[0]), float(α[-1]), float(β[0]),  float(β[-1]) ], aspect = 'auto')
        cb = fig.colorbar(im, orientation = 'vertical',fraction=0.045)
        cb.set_label(label = r'$\frac{\partial P(α, β)}{\partial \Omega \cos(\theta)}$ [a.u.]', size = 12)
        ax.set_title(r"Target radiant intensity per cosine")
        ax.set_aspect(1) 
        plt.show()
