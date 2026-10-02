from __future__ import annotations
import matplotlib.pyplot as plt

class MoldesLivePlot:
    """Visualização em tempo real do ambiente moldes_fmu. """

    def __init__(self) -> None:
        self.times = []
        self.temperatures = []
        self.references = []
        self.water_flows = []
        self.mold_states = []

        plt.ion()

        self.fig, self.axes = plt.subplots(3, 1, figsize=(9, 7))
        self.fig.suptitle("FMU Environment-Injection Moulds")

        self.ax_temp = self.axes[0]
        self.ax_flow = self.axes[1]
        self.ax_state = self.axes[2]

    def _fix_temperature_for_plot(self, value: float) -> float:
        """
        Corrige apenas para visualização.

        Se aparecer -250 ºC, significa que foi feito -273.15
        a um valor que já estava em ºC.
        """
        if value < -100.0:
            return value + 273.15

        return value

    def update(
        self,
        time: float,
        temperature_celsius: float,
        reference_celsius: float,
        water_flow: float,
        mold_state: float,
    ) -> None:

        temperature_plot = self._fix_temperature_for_plot(temperature_celsius)
        reference_plot = self._fix_temperature_for_plot(reference_celsius)

        self.times.append(time)
        self.temperatures.append(temperature_plot)
        self.references.append(reference_plot)
        self.water_flows.append(water_flow)
        self.mold_states.append(mold_state)

        self.ax_temp.clear()
        self.ax_flow.clear()
        self.ax_state.clear()

        # Temperatura do molde
        self.ax_temp.plot(
            self.times,
            self.temperatures,
            label="T1",
        )

        self.ax_temp.plot(
            self.times,
            self.references,
            linestyle="--",
            label="Reference",
        )

        self.ax_temp.set_ylim(0, 50)
        self.ax_temp.set_ylabel("Temperature [ºC]")
        self.ax_temp.legend()
        self.ax_temp.grid(True)

        # Caudal de água
        self.ax_flow.plot(
            self.times,
            self.water_flows,
            color="yellow",
            label="mH2O",
        )

        self.ax_flow.set_ylabel("flow rate [kg/s]")
        self.ax_flow.legend()
        self.ax_flow.grid(True)

        # Estado dos moldes
        if len(self.times) >= 2:
            for i in range(1, len(self.times)):
                state = self.mold_states[i - 1]

                color = "green" if state == 0 else "red"

                self.ax_state.step(
                    self.times[i - 1:i + 1],
                    self.mold_states[i - 1:i + 1],
                    where="post",
                    color=color,
                )
        else:
            color = "green" if mold_state == 0 else "red"
            self.ax_state.step(
                self.times,
                self.mold_states,
                where="post",
                color=color,
            )

        self.ax_state.set_ylabel("Moulds")
        self.ax_state.set_xlabel("Time [s]")
        self.ax_state.set_yticks([0, 1])
        self.ax_state.set_yticklabels(["Open", "Closed"])
        self.ax_state.grid(True)

        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)