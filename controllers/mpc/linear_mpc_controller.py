from __future__ import annotations

from typing import Any, Dict, Optional

import cvxpy as cp
import numpy as np

from controllers.mpc.base_mpc_controller import BaseMPCController


class LinearMPCController(BaseMPCController):
    """
    Controlador MPC linear/afim.

    Suporta modelos do tipo:

        x(k+1) = A x(k) + B u(k)

    e também:

        x(k+1) = A x(k) + B u(k) + C s(k) + d

    Isto permite usar modelos identificados a partir de FMUs.
    """

    def __init__(
        self,
        name: str,
        mpc_type: str,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        prediction_horizon: int = 10,
        control_horizon: Optional[int] = None,
        reference: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, Any]] = None,
        solver: Optional[str] = None,
        C: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        dt: float = 1.0,
        cycle_period: Optional[float] = None,
        cycle_open_time: Optional[float] = None,
    ) -> None:

        model = {
            "A": A,
            "B": B,
            "Q": Q,
            "R": R,
            "C": C,
            "d": d,
        }

        super().__init__(
            name=name,
            mpc_type=mpc_type,
            config=config,
            model=model,
            prediction_horizon=prediction_horizon,
            control_horizon=control_horizon,
            reference=reference,
            constraints=constraints,
            solver=solver,
        )

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.C = None if C is None else np.asarray(C, dtype=float)
        self.d = None if d is None else np.asarray(d, dtype=float).reshape(-1)

        self.dt = float(dt)
        self.cycle_period = cycle_period
        self.cycle_open_time = cycle_open_time

        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

        if self.reference is None:
            self.reference = np.zeros(self.nx)

        self._time = 0.0

    def reset(self) -> None:
        """
        Reinicia o controlador entre episódios.
        """
        super().reset()
        self._time = 0.0

    def _get_s_value(self, t: float) -> float:
        """
        Calcula o estado previsto dos moldes.

        s = 0 -> moldes abertos
        s = 1 -> moldes fechados
        """

        if self.cycle_period is None or self.cycle_open_time is None:
            return 0.0

        t_cycle = t % float(self.cycle_period)

        if t_cycle < float(self.cycle_open_time):
            return 0.0

        return 1.0

    def solve_mpc(self, observation: Any) -> Any:
        """
        Resolve o problema MPC e devolve a primeira ação ótima.
        """

        observation = np.asarray(observation, dtype=float).reshape(-1)

        # O modelo identificado dos moldes usa apenas a temperatura
        # como estado do MPC.
        # Caso a observação contenha variáveis extra (ex: s_out),
        # elas são ignoradas para o estado inicial.
        x0 = observation[: self.nx]

        x_ref = np.asarray(
            self.reference,
            dtype=float,
        ).reshape(-1)

        Np = self.prediction_horizon
        Nc = self.control_horizon

        x = cp.Variable((self.nx, Np + 1))
        u = cp.Variable((self.nu, Nc))

        cost = 0
        constraints = []
        
        if len(x0) != self.nx:
            raise ValueError(
                f"Estado recebido com dimensão {len(x0)} "
                f"mas o modelo espera {self.nx} estados."
            )
        constraints.append(x[:, 0] == x0)

        u_min = self.constraints.get("u_min", None)
        u_max = self.constraints.get("u_max", None)
        x_min = self.constraints.get("x_min", None)
        x_max = self.constraints.get("x_max", None)

        for k in range(Np):
            if k < Nc:
                uk = u[:, k]
            else:
                uk = u[:, Nc - 1]

            # Custo de seguimento da referência.
            cost += cp.quad_form(x[:, k] - x_ref, self.Q)

            # Custo do esforço de controlo.
            cost += cp.quad_form(uk, self.R)

            # Dinâmica base: A x + B u.
            next_x = self.A @ x[:, k] + self.B @ uk

            # Termo identificado C*s(k), se existir.
            if self.C is not None:
                t_pred = self._time + k * self.dt
                s_k = self._get_s_value(t_pred)
                next_x = next_x + (self.C.reshape(self.nx) * s_k)

            # Termo constante identificado d, se existir.
            if self.d is not None:
                next_x = next_x + self.d

            constraints.append(x[:, k + 1] == next_x)

            if u_min is not None:
                constraints.append(uk >= u_min)

            if u_max is not None:
                constraints.append(uk <= u_max)

            if x_min is not None:
                constraints.append(x[:, k] >= x_min)

            if x_max is not None:
                constraints.append(x[:, k] <= x_max)

        cost += cp.quad_form(x[:, Np] - x_ref, self.Q)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=self.solver)

        if u.value is None:
            raise RuntimeError(f"[{self.name}] O problema MPC não foi resolvido.")

        action = np.asarray(u.value[:, 0]).reshape(-1)

        # Atualiza o tempo interno do controlador.
        self._time += self.dt

        if action.size == 1:
            return float(action[0])

        return action

    def get_info(self) -> Dict[str, Any]:
        """
        Devolve informação detalhada do controlador Linear MPC.
        """
        info = super().get_info()
        info.update(
            {
                "nx": self.nx,
                "nu": self.nu,
                "has_Q": self.Q is not None,
                "has_R": self.R is not None,
                "has_C": self.C is not None,
                "has_d": self.d is not None,
                "dt": self.dt,
                "cycle_period": self.cycle_period,
                "cycle_open_time": self.cycle_open_time,
            }
        )
        return info