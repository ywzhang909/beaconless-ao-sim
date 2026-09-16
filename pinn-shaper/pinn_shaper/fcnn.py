import torch
import torch.nn as nn

"""
A small implementation of a *fully‑connected neural network* (also
known as a multilayer perceptron) with an **optional hard constraint** applied
to its output.  The design targets physics‑informed neural networks (PINNs),
where one often wants the network to exactly satisfy certain known conditions –
for instance boundary values or conservation laws – while the trainable part
(``f``) learns the remaining degrees of freedom.
"""

class FCNN(nn.Module):

    """Fully‑connected neural network with an optional output constraint.


    * ``f(x)`` – an unconstrained neural approximation (initialised close to
      zero and scaled by a learnable **scalar** ``f_scale``), and
    * ``g(x)`` – a user‑supplied *analytical* term that encodes known physics
      (default: squared radius *r²*).

    The final output is

    .. math::

        f(x) = f(x) + g(x),

    where the second term is included **only if** ``hard_constrain`` is
    ``True``.

    Parameters
    ----------
    input_dim : int, default=2
        Spatial dimension of the input vector (x, y)

    hidden_layers : int, default=3
        Number of *hidden* ``Linear → Tanh`` blocks.  Must be ≥1.
    num_features : int, default=150
        Width of every hidden layer.
    output_dim : int, default=1
        Size of the network output. 
    hard_constrain : bool, default=False
        If *True, the *analytical* term ``constrain_fn`` is added to the network
        output *f*.  If *False*, the model degenerates to a standard MLP.
    constrain_fn : Callable[[Tensor], Tensor] | None, optional
        A function :math:`g(x)` whose **output shape** must match
        ``output_dim``.  The default returns
        .. math::
            g(x) = x_0^2 + x_1^2
    """

    def __init__(self, input_dim=2, hidden_layers=3, num_features=150,
                 output_dim=1, hard_constrain=False, constrain_fn=None):
        super().__init__()
        layers = [nn.Linear(input_dim, num_features), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(num_features, num_features), nn.Tanh()]
        layers.append(nn.Linear(num_features, output_dim))

        # the learnable, *unconstrained* part of the model -----------------
        self.net  = nn.Sequential(*layers)

        # Hard‑constraint bookkeeping -----------------------------------------
        self.hard = hard_constrain
        self.f_scale = nn.Parameter(torch.tensor(1.0))

        # default constraint: r²
        self.constrain_fn = (constrain_fn if constrain_fn is not None
                             else lambda x: x[:, 0:1]**2 + x[:, 1:2]**2)

    def forward(self, x):
        """Compute *f(x)*.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(*, input_dim)`` where ``*`` is an arbitrary batch
            prefix.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(*, output_dim)``.
        """

        f = self.f_scale * self.net(x)
        return f + self.constrain_fn(x) if self.hard else f
