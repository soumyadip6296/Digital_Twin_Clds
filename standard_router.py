"""
standard_router.py — Simulates a traditional (non-AI) router.

Three selectable modes for the head-to-head comparison:

  static  Always uses Primary link. Never adapts to conditions.
          Represents the "do nothing" baseline.

  random  Randomly picks Primary or Backup per packet.
          Represents the worst-case "no intelligence" baseline.

  ospf    Reacts to congestion AFTER it is detected — mimics OSPF
          link-state convergence. Switches only after N consecutive
          high-latency packets (convergence delay), and restores
          Primary only after it has been stable again.
"""

import numpy as np
from collections import deque
from typing import Literal, Dict, Any

RouterMode = Literal["ospf", "static", "random"]


class StandardRouter:
    # OSPF convergence parameters
    OSPF_HIGH_THRESHOLD  = 0.40   # lat_a above this = congested
    OSPF_RECOVERY_THRESHOLD = 0.10   # lat_a below this = recovered
    OSPF_TRIGGER_N       = 3      # consecutive high-lat packets before failover

    def __init__(self, mode: RouterMode = "ospf"):
        self.mode: RouterMode = mode
        self._route:      int   = 0        # 0 = Primary, 1 = Backup
        self._cong_count: int   = 0        # rising-congestion counter
        self._lat_buf:    deque = deque(maxlen=5)

        # cumulative score tracking
        self.packets_decided:    int   = 0
        self.total_latency:      float = 0.0
        self.late_reactions:     int   = 0   # packets where router was on wrong link
        self.route_switches:     int   = 0

    # ------------------------------------------------------------------
    def decide(
        self,
        lat_a: float,
        lat_b: float,
        volume: float = 0.0,
        error_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Returns
        -------
        route               int   0 = Primary, 1 = Backup
        reason              str   human-readable explanation
        experienced_latency float latency the packet actually encounters
        reacted_late        bool  True when the router is on Primary during congestion
                                  (the dangerous window — packet suffers high latency)
        """
        prev_route = self._route

        if self.mode == "static":
            out = self._static(lat_a)
        elif self.mode == "random":
            out = self._random(lat_a, lat_b)
        else:
            out = self._ospf(lat_a, lat_b)

        # cumulative stats
        if self._route != prev_route:
            self.route_switches += 1
        self.packets_decided += 1
        self.total_latency   += out["experienced_latency"]
        if out["reacted_late"]:
            self.late_reactions += 1

        return out

    # ------------------------------------------------------------------
    def _static(self, lat_a: float) -> Dict[str, Any]:
        self._route = 0
        return {
            "route": 0,
            "reason": "Static: Always Primary — no adaptation logic.",
            "experienced_latency": lat_a,
            "reacted_late": lat_a > self.OSPF_HIGH_THRESHOLD,
        }

    def _random(self, lat_a: float, lat_b: float) -> Dict[str, Any]:
        r = int(np.random.randint(0, 2))
        self._route = r
        lat = lat_b if r == 1 else lat_a
        return {
            "route": r,
            "reason": f"Random: Coin-flip → {'Backup' if r else 'Primary'}.",
            "experienced_latency": lat,
            "reacted_late": (r == 0 and lat_a > self.OSPF_HIGH_THRESHOLD),
        }

    def _ospf(self, lat_a: float, lat_b: float) -> Dict[str, Any]:
        self._lat_buf.append(lat_a)
        avg = float(np.mean(self._lat_buf))

        if self._route == 0:                           # currently Primary
            if avg > self.OSPF_HIGH_THRESHOLD:
                self._cong_count += 1
                if self._cong_count >= self.OSPF_TRIGGER_N:
                    self._route = 1
                    reason = (
                        f"OSPF: Congestion confirmed after {self.OSPF_TRIGGER_N} packets "
                        f"(avg_lat={avg:.3f}) → Failover to Backup"
                    )
                else:
                    reason = (
                        f"OSPF: Congestion building "
                        f"({self._cong_count}/{self.OSPF_TRIGGER_N}) — "
                        f"still on Primary (avg_lat={avg:.3f})"
                    )
            else:
                self._cong_count = 0
                reason = f"OSPF: Primary healthy (avg_lat={avg:.3f})"
        else:                                          # currently Backup
            if avg < self.OSPF_RECOVERY_THRESHOLD:
                self._route = 0
                self._cong_count = 0
                reason = f"OSPF: Primary recovered (avg_lat={avg:.3f}) → Restored"
            else:
                reason = f"OSPF: Remaining on Backup (avg_lat={avg:.3f})"

        lat = lat_b if self._route == 1 else lat_a
        reacted_late = (self._route == 0 and lat_a > self.OSPF_HIGH_THRESHOLD)
        return {
            "route": self._route,
            "reason": reason,
            "experienced_latency": lat,
            "reacted_late": reacted_late,
        }

    # ------------------------------------------------------------------
    @property
    def avg_latency(self) -> float:
        if self.packets_decided == 0:
            return 0.0
        return self.total_latency / self.packets_decided

    @property
    def late_reaction_pct(self) -> float:
        if self.packets_decided == 0:
            return 0.0
        return 100.0 * self.late_reactions / self.packets_decided

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "packets": self.packets_decided,
            "avg_latency": round(self.avg_latency, 4),
            "late_reactions": self.late_reactions,
            "late_reaction_pct": round(self.late_reaction_pct, 2),
            "route_switches": self.route_switches,
        }

    def reset(self):
        self._route      = 0
        self._cong_count = 0
        self._lat_buf.clear()
        self.packets_decided  = 0
        self.total_latency    = 0.0
        self.late_reactions   = 0
        self.route_switches   = 0

    def set_mode(self, mode: RouterMode):
        self.mode = mode
        self.reset()
