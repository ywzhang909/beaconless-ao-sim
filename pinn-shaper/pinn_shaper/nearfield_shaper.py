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

class Nearfield_PINN:

    """Physics‑informed NN for designing phase profiles for flat-optics devices.
    Given a desired complex field distribution at a *target* plane `z` and a
    known intensity profile at the *source* plane, the class trains a fully
    connected neural network Phi(x, y) that parametrises an unknown phase
    profile.

    Parameters
    ----------
    I_fun, E_fun
        Callables returning *source irradiance* I(X, Y) and *desired
        target irradiance* :E(X, Y) respectively.  Each must accept two
        broadcast‑compatible tensors (X, Y) and return a tensor of the
        same shape.
    extent_source, extent_target : float
        Physical width of the *source* and *target*
        planes, **in the same units**.  
    z : float
        Axial distance from source to target plane.
    λ : float
        Wavelength 
    phi_scale : float, default=30
        Scaling hyperparameter applied to the learnable phase network before 
        entering the PDE.
    integration_points : int, default=5000
        Number of points for integration when normalising the
        supplied irradiances such that total power in both planes is identical.
    hidden_layers, num_features
        Architecture of the underlying :class:`fcnn.FCNN`.
    constrain : bool, default=False
        Forwarded to :class:`fcnn.FCNN`; add an *exact* hard constraint to the
        network output.
    symmetry : {``'x'``, ``'y'``, ``'xy'``} or ``None``
        Reduces the solved domain by a factor of 2 or 4 when physical symmetry
        is present (e.g. circular aperture).
        Geometric symmetry reduction; chooses a quarter‑, half‑ or full domain.
    constrain_fn : Callable[[Tensor], Tensor], optional
        Analytical function *g(x)* to include in the hard constraint path.
    """


    def __init__(self, I_fun, E_fun, extent_source, extent_target, z, λ, phi_scale = 30,
                 integration_points=5000, hidden_layers=3, num_features=150,
                 constrain=False,  symmetry=None, constrain_fn=None):

        # -------- domain & symmetry -----------------------------------
        sym = None if symmetry is None else symmetry.lower()
        if   sym == 'xy':
            self.x_range, self.y_range = (0, 1), (0, 1)
        elif sym == 'x':
            self.x_range, self.y_range = (0, 1), (-1, 1)
        elif sym == 'y':
            self.x_range, self.y_range = (-1, 1), (0, 1)
        elif sym is None:
            self.x_range, self.y_range = (-1, 1), (-1, 1)
        else:
            raise ValueError("symmetry must be 'x', 'y', 'xy', or None")
        self.symmetry = sym

        # -------------- rest of constructor -------------------------------
        # (normalise E_fun, build network, etc.)
        # ------------------------------------------------------------------
        # ---------------- optical constants & scaling factors ---------
        self.M = extent_source / 2.0  # half‑width of the *source* plane
        self.T = extent_target / 2.0  # half‑width of the *target* plane
        self.z = z                    # propagation distance
        self.λ = λ                    # wavelength
        self.s = self.M / z           # *reduced* distance – appears in PDE
        self.phi_scale = phi_scale    # scaling hyperparameter

        # ---------------- power normalisation -------------------------
        N = integration_points
        dx = dy = 2.0 / N
        x_bar = dx * (torch.arange(N) - N//2)
        y_bar = dy * (torch.arange(N) - N//2)
        xx_bar, yy_bar = torch.meshgrid(x_bar, y_bar, indexing='xy')

        I_vals = I_fun(self.M*xx_bar, self.M*yy_bar)
        E_vals = E_fun(self.T*xx_bar, self.T*yy_bar)

        # 2‑D trapezoidal rule – accurate enough for smooth irradiances.
        area_I = torch.trapz(torch.trapz(I_vals, x=self.M*x_bar, dim=1), x=self.M*y_bar, dim=0)
        area_E = torch.trapz(torch.trapz(E_vals, x=self.T*x_bar, dim=1), x=self.T*y_bar, dim=0)

        self.area_ratio = area_I / area_E # ensures power conservation
        self.E_fun_norm = lambda X, Y: E_fun(X, Y) * self.area_ratio
        self.I_fun      = I_fun

        # ---------------- neural field --------------------------------
        self.net = FCNN(2, hidden_layers, num_features, 1,
                        constrain, constrain_fn)
        self.net.train()

    # --------- sampling helpers ----------
    def _grid(self, n, rng): 
        """Return a 1‑D tensor with *n* uniformly spaced samples in *rng*."""
        return torch.linspace(rng[0], rng[1], n)

    def sample_interior_points(self, N, noise=0.001):
        """Quasi‑uniform sampling of interior domain points.

        Uses a *√N × √N* Cartesian grid, cropped to *N* points, then adds small
        Gaussian jitter (`noise`) to avoid aliasing artefacts in the PDE loss.
        """
        m=int(np.ceil(np.sqrt(N)))
        X,Y=torch.meshgrid(self._grid(m,self.x_range),
                           self._grid(m,self.y_range),indexing='xy')
        pts=torch.stack([X.reshape(-1),Y.reshape(-1)],1)[:N]
        return pts+noise*torch.randn_like(pts) if noise else pts

    # ---- boundary samplers (for symmetry‑induced Neumann BCs) --------
    def sample_boundary_x(self, N, noise=0.001):
        """Points on the *x* = constant boundary (vertical edge)."""
        if self.symmetry not in ('x', 'xy'):
            return torch.empty(0,2)
        y=self._grid(N,self.y_range).unsqueeze(1)
        pts=torch.cat([torch.full_like(y, self.x_range[0]), y],1)
        return pts+noise*torch.randn_like(pts) if noise else pts

    def sample_boundary_y(self, N, noise=0.001):
        """Points on the *y* = constant boundary (horizontal edge)."""
        if self.symmetry not in ('y', 'xy'):
            return torch.empty(0,2)
        x=self._grid(N,self.x_range).unsqueeze(1)
        pts=torch.cat([x, torch.full_like(x, self.y_range[0])],1)
        return pts+noise*torch.randn_like(pts) if noise else pts



    # ------------- PDE residual ----------------------------------------
    def pde_residual(self, x, power):
        """
        Point‑wise residual for the near‑field ray‑mapping Monge–Ampère PDE.

        Full PDE (normalised units)
        ---------------------------
            E(t_x, t_y) · [ Φ_xx · Φ_yy − Φ_xy²  + K · (1 − s² Φ_y²) · Φ_xx 
            + K · (1 − s² Φ_x²) · Φ_yy + 2 s² K Φ_x Φ_y Φ_xy  + K⁴]  =  K⁴ · I(x, y)

        with the auxiliary definitions

            K   = sqrt( 1 − s² (Φ_x² + Φ_y²) )        # local ray‑compression factor
            t_x = x + Φ_x / K                         # mapped target x‑coord
            t_y = y + Φ_y / K                         # mapped target y‑coord
            s   = M / z                               # reduced distance,  M = extent_source / 2

        • I(x,y)  – source‑plane irradiance (self.I_fun).  
        • E(t_x,t_y) – power‑normalised target irradiance (self.E_fun_norm).  
        • Φ_x, Φ_y, Φ_xx, …  – first‑ and second‑order derivatives of the phase
          potential Φ predicted by the neural network.

        The method returns |LHS−RHS| raised to the given exponent *power*.
        A mean over these residuals forms the interior‑loss term used during
        training.

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
        Φ     = self.net(x) / self.phi_scale
        
        # 1st‑order derivatives -------------------------------------------------
        grad  = torch.autograd.grad(Φ, x, torch.ones_like(Φ), create_graph=True)[0]
        Φx, Φy = grad[:, 0:1], grad[:, 1:2]

        # 2nd‑order derivatives -------------------------------------------------
        Φxx = torch.autograd.grad(Φx, x, torch.ones_like(Φx), create_graph=True)[0][:, 0:1]
        Φyy = torch.autograd.grad(Φy, x, torch.ones_like(Φy), create_graph=True)[0][:, 1:2]
        Φxy = torch.autograd.grad(Φx, x, torch.ones_like(Φx), create_graph=True)[0][:, 1:2]

        # auxiliary variables --------------------------------------------------
        Kbar = torch.sqrt(torch.clamp(1 - self.s**2 * (Φx**2 + Φy**2), min=1e-12))
        tx   = x[:, 0:1] + Φx / Kbar
        ty   = x[:, 1:2] + Φy / Kbar

        # coefficients of the fully‑non‑linear Monge–Ampère operator ----------
        A1 = 1.0
        A2 = Kbar * (1 - self.s**2 * Φy**2)
        A3 = Kbar * (1 - self.s**2 * Φx**2)
        A4 = 2 * self.s**2 * Kbar * Φx * Φy
        A5 = Kbar**4

        det_term = Φxx * Φyy - Φxy**2
        NL_expr  = (A1 * det_term + A2 * Φxx + A3 * Φyy + A4 * Φxy + A5)

        lhs = self.E_fun_norm(self.M * tx, self.M * ty) * torch.abs(NL_expr)
        rhs = (Kbar**4) * self.I_fun(self.M * x[:, 0:1], self.M * x[:, 1:2])

        return torch.abs(lhs - rhs)**power

    # ==================================================================
    # boundary losses – enforce *Neumann* symmetry BCs (∂Φ/∂n = 0)
    # ==================================================================
    def boundary_loss_x(self, xb):
        """Gradient along *x* for points on the x‑edge (symmetry plane)."""
        if self.symmetry != 'xy' and self.symmetry != 'x':
            return torch.zeros(1, device=xb.device)
        if xb.numel() == 0:
            return torch.zeros(1, device=xb.device)
        xb.requires_grad_(True)
        grad = torch.autograd.grad(self.net(xb), xb,
                                   torch.ones_like(xb[:, :1]),
                                   create_graph=True)[0]
        return grad[:, 0:1]

    def boundary_loss_y(self, xb):
        """Gradient along *y* for points on the y‑edge (symmetry plane)."""
        if self.symmetry != 'xy' and self.symmetry != 'y':
            return torch.zeros(1, device=xb.device)
        if xb.numel() == 0:
            return torch.zeros(1, device=xb.device)
        xb.requires_grad_(True)
        grad = torch.autograd.grad(self.net(xb), xb,
                                   torch.ones_like(xb[:, :1]),
                                   create_graph=True)[0]
        return grad[:, 1:2]

    # ------------- test-loss helper ------------------------------------
    def compute_test_loss(self, n_int, n_bnd, pwr, w, noise):
        """Return *(total, interior, BCx, BCy)* test loss values.*"""
        x_int = self.sample_interior_points(n_int, noise)
        x_bx  = self.sample_boundary_x(n_bnd, noise)
        x_by  = self.sample_boundary_y(n_bnd, noise)

        L_int = self.pde_residual(x_int, pwr).mean()
        L_bx  = self.boundary_loss_x(x_bx).pow(2).mean() if x_bx.numel() else torch.tensor(0.)
        L_by  = self.boundary_loss_y(x_by).pow(2).mean() if x_by.numel() else torch.tensor(0.)
        total = w[0]*L_int + w[1]*L_bx + w[2]*L_by
        return total.item(), L_int.item(), L_bx.item(), L_by.item()

    # ==================================================================
    #  Adam phase
    # ==================================================================
    def run_phase_adam(self, *, iterations, num_domain, num_boundary,
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

        opt = optim.Adam(self.net.parameters(), lr=lr)
        for it in range(iterations):
            opt.zero_grad()
            xD = self.sample_interior_points(num_domain, effective_noise)
            xBx= self.sample_boundary_x(num_boundary, effective_noise)
            xBy= self.sample_boundary_y(num_boundary, effective_noise)

            L_int= self.pde_residual(xD, residual_power).mean()
            L_bx = self.boundary_loss_x(xBx).pow(2).mean() if xBx.numel() else torch.tensor(0.)
            L_by = self.boundary_loss_y(xBy).pow(2).mean() if xBy.numel() else torch.tensor(0.)
            loss = loss_weights[0]*L_int + loss_weights[1]*L_bx + loss_weights[2]*L_by
            loss.backward(); opt.step()

            if it % 100 == 0:
                T, Li, Lbx, Lby = self.compute_test_loss(
                    num_test_interior, num_test_boundary,
                    residual_power, loss_weights, effective_noise)
                print(f"Adam Iter {it:5d}: Loss = {loss.item():.8f} "
                      f"(Interior = {L_int.item():.8f}, BC_x = {L_bx.item():.8f}, BC_y = {L_by.item():.8f})")
                print(f"           Test Loss = {T:.8f} "
                      f"(Interior = {Li:.8f}, BC_x = {Lbx:.8f}, BC_y = {Lby:.8f})")

    # ==================================================================
    #  LBFGS phase
    # ==================================================================
    def run_phase_lbfgs(self, *, num_domain, num_boundary, residual_power,
                        loss_weights, effective_noise,
                        num_test_interior, num_test_boundary,
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

        xD  = self.sample_interior_points(num_domain, effective_noise)
        xBx = self.sample_boundary_x(num_boundary, effective_noise)
        xBy = self.sample_boundary_y(num_boundary, effective_noise)
        count = [0]

        def closure():
            opt.zero_grad()
            Li  = self.pde_residual(xD, residual_power).mean()
            Lbx = self.boundary_loss_x(xBx).pow(2).mean() if xBx.numel() else torch.tensor(0.)
            Lby = self.boundary_loss_y(xBy).pow(2).mean() if xBy.numel() else torch.tensor(0.)
            loss = loss_weights[0]*Li + loss_weights[1]*Lbx + loss_weights[2]*Lby
            loss.backward()
            count[0] += 1
            if count[0] % 100 == 0:
                T, Li_, Lbx_, Lby_ = self.compute_test_loss(
                    num_test_interior, num_test_boundary,
                    residual_power, loss_weights, effective_noise)
                print(f"LBFGS Iter {count[0]:5d}: Loss = {loss.item():.8f} "
                      f"Test Loss = {T:.8f} (Interior = {Li_:.8f}, "
                      f"BC_x = {Lbx_:.8f}, BC_y = {Lby_:.8f})")
            return loss

        opt = optim.LBFGS(self.net.parameters(), max_iter=max_iterations_lbfgs,
                          lr=lr_lbfgs, line_search_fn="strong_wolfe")
        opt.step(closure)

    def run_phase(self, optimizer_type, **kw):
        """Thin wrapper for ``run_phase_adam`` / ``run_phase_lbfgs``."""
        if optimizer_type.lower() == 'adam':
            self.run_phase_adam(**kw)
        elif optimizer_type.lower() == 'lbfgs':
            self.run_phase_lbfgs(**kw)
        else:
            raise ValueError("optimizer_type must be 'adam' or 'lbfgs'")
            
    def print_final_test_loss(self, num_test_interior, num_test_boundary, residual_power, loss_weights, effective_noise):
        """Pretty‑print the final held‑out loss after training.

        This is a convenience wrapper around
        :py:meth:`compute_test_loss <Nearfield_PINN.compute_test_loss>` that
        formats the returned values in a human‑readable way.

        Parameters
        ----------
        num_test_interior, num_test_boundary : int
            Point counts for the *interior* and *boundary* evaluation sets.
        residual_power : float
            Exponent *p* used in the PDE residual, must match training.
        loss_weights : sequence[float]
            Three weights ``(w_int, w_bx, w_by)`` combining the loss terms.
        effective_noise : float
            Jitter added to evaluation points to avoid aliasing artefacts when
            deterministic sampling grids are used.
        """

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

            Phi(x, y) = FCNN(x, y) * k  * M^2 / z / phi_scale  

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
            phi_pred = self.net((XY/self.M)) * k  * (self.M**2) /  self.z     /(self.phi_scale)  
            
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





    def plot_target_irradiance(self, figsize=(5, 4),  units = mm, grid = False, res = (2048,2048)):
        """Render the *target*‑plane irradiance $E(x, y)$ (power‑normalised).

        Identical interface to :py:meth:`plot_source_irradiance`; only the
        plotted function differs (``E_fun_norm`` instead of ``I_fun``) and the
        spatial range is ``[-T, T]``.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if grid == True:
            ax.grid(alpha =0.2)

        self._configure_axes_unit_labels(ax, units)

        nx, ny = res
        x = torch.linspace(-self.T, self.T, nx)
        y = torch.linspace(-self.T, self.T, ny)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        im = ax.imshow(self.E_fun_norm(X, Y).cpu().numpy(), origin = 'lower', cmap ='inferno', extent = [float(x[0])/units, float(x[-1])/units, float(y[0])/units,  float(y[-1])/units ], aspect = 'auto')
        cb = fig.colorbar(im, orientation = 'vertical',fraction=0.045, label = 'Intensity [a.u.]')
        ax.set_title(r"Target plane irradiance $E(x,y)$")
        ax.set_aspect(1) 
        plt.show()
