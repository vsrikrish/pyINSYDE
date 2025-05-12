#%%

import os
import math
from dataclasses import dataclass, fields
import numpy as np
from scipy.stats import truncnorm, norm


"""pyINSYDE: Flood damage simulation.

Python port of the INSYDE model from Dottori et al (2016). This module contains code to determine water levels and damages.
"""

# read replacement value and unit cost data
rep_val_data = np.genfromtxt("data/replacement_values.txt", delimiter=" ", skip_header=2, usecols=(0, 1, 2))

uc_lines = []                                                     
with open('data/unit_prices.txt', 'r') as f: 
    next(f)
    next(f)
    for line in f:
        if not line.lstrip().startswith('#'):
            uc_lines.append(line)   
strip_list = [line.replace('\n','').split() for line in uc_lines if line != '\n']
unit_cost_data = dict()
for line in strip_list: 
    unit_cost_data[line[0]] = float(line[1]) 

@dataclass
class BuildingProperties:
    """Properties for a building class.

    Attributes:
        FA: footprint area (m^2).
        IA: internal area (m^2); default is 0.9 * FA if missing.
        BA: basement area (m^2); default is 0.5 * FA if missing.
        EP: external perimeter (m).
        IH: interstory height (m).
        BH: basement height (m).
        GL: ground floor level (m); default is 0.
        NF: number of floors.
        BT: building type; 1 - detached; 2 - semi-detached; 3 - apartment house.
        BS: building structure; 1 - reinforced concrete, 2 - masonry, 3 - wood.
        PD: plant distribution; 1 - centralized, 2 - distributed.
        PT: heating system type; 1 - radiator; 2 - underfloor heating.
        FL: finishing level coefficient; 0.8 is low, 1 medium (default), 1.2 high.
        YY: year of construction.
        LM: level of maintanance; 0.9 low, medium 1 (default), 1.1 high.
        sd: standard deviation of perturbations to fragility curves; default = 0 (no uncertainty).
    """
    FA: float
    IA: float = None
    BA: float = None
    EP: float
    IH: float
    BH: float
    GL: float = 0.0
    NF: int
    BT: int
    BS: int
    PD: int
    PT: int
    FL: float = 1.0
    YY: float
    LM: float = 1.1
    sd: float = 0.0

    def items(self):
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)

class Building:
    """Building with INSYDE-relevant characteristics.

    This class holds structural characteristics for a given building,
    which are then used to calculate water levels and damages.

    Attributes:
    
    """

    def __init__(self, building_data:BuildingProperties):
        """Initializes instance based on dictionary of characteirstics.

        Args:
            prop: BuildingProperties type containing non-default data.
        """
        for k, v in building_data.items():
            setattr(self, k, v)
        if self.IA is None:
            self.IA = 0.9 * self.FA
        if self.BA is None:
            self.BA = 0.5 * self.FA

        # get replacement value
        self.repVal = rep_val_data[self.BS - 1, self.BT - 1]
        # get unit costs
        self.costs = unit_cost_data     

        # compute exposure variables
        self.IP = 2.5 * self.EP  # Internal perimeter (m)
        self.BP = 4 * np.sqrt(self.BA)  # Basement perimeter (m)
        self.BL = self.GL - 0.3 - self.BH  # Basement level (m)

        # Calculate replacement values (new and used)
        self.RVN = self.repVal * self.FA * self.NF  
        age = 2015 - self.YY
        decay = min(0.01 * age / self.LM, 0.3)
        self.RVU = self.RVN * (1 - decay)


    def waterLevel(self, he):
        """
        Based on lines 7-11 INSYDE, depends on:
        - he: water depth outside of the building (m)
        - NF: number of floors 
        - IH: interfloor height (m)
        - GL: ground floor level (m)

        Returns:
        - h: water depth inside the building for each floor (m)
        More description in Tables 2 and 3
        """
        # Calculate the water depth at the ground level

        h = np.minimum(he, self.NF * self.IH * 1.05)    
        depth = np.round(h - self.GL, 3)    
        depth = np.where(depth > 0, depth, 0)
        return depth

    def fragility(self, 
        he:float, 
        v:float, 
        d:float
        ) -> tuple:
        """Fragility curves from flooding.

        Args:
            he: water depth outside of the building (m).
            v: velocity (m/s).
            d = flood duration (h).
        """

        def ptruncnorm(x, a, b, mean, sd, sd_perturb=self.sd):
            lower, upper = (a - mean) / sd, (b - mean) / sd 
            x_perturb = x + norm.rvs(0, sd_perturb * sd)
            return truncnorm.cdf(x_perturb, lower, upper, loc=mean, scale=sd)
        
        depth = self.waterlevel(he)
        # damage due to flood duration
        # starts at 12 hours and maximizes after 36
        frag1 = np.round(ptruncnorm(d, a=12, b=36, mean=24, sd=24/6), 3)
        # damage due to wood floor damage (0.2-0.6m)
        frag2_1f = np.round(ptruncnorm(depth, a=0.2, b=0.6, mean=0.4, sd=0.4/6), 3)
        frag2_2f = np.round(ptruncnorm(depth, a=0.2, b=0.6, mean=0.4, sd=0.4/6), 3)
        # damage to partitions (1.5-2m)
        frag3_1f = np.round(ptruncnorm(depth, a=1.5, b=2.0, mean=1.75, sd=0.5/6), 3)
        frag3_2f = np.round(ptruncnorm(depth, a=1.5 + self.IH, b=2.0 + self.IH, mean=1.75 + self.IH, sd=0.5/6), 3)       
        # damage to external plaster and doors (1-1.5m)
        frag4 = np.round(ptruncnorm(v, a=1, b=1.5, mean=1.25, sd=0.5/6), 3)
        # damage to doors (0.4 - 0.8 m)
        frag5_1f = np.round(ptruncnorm(depth, a=0.4, b=0.8, mean=0.6, sd=0.4/6), 3)
        frag5_2f = np.round(ptruncnorm(depth, a=0.4 + self.IH, b=0.8 + self.IH, mean=0.6 + self.IH, sd=0.4/6), 3)
        # damage to windows (1.2 - 1.8 m)
        frag6_1f = np.round(ptruncnorm(depth, a=1.2, b=1.8, mean=1.5, sd=0.5/6), 3)
        frag6_2f = np.round(ptruncnorm(depth, a=1.2 + self.IH, b=1.8 + self.IH, mean=1.5 + self.IH, sd=0.5/6), 3)
        # damage to windows (0.8 - 1.0 m/s)
        frag7 = np.round(ptruncnorm(v, a=0.8, b=1.0, mean=0.9, sd=0.2/6), 3)
        # structural damage
        frag8 = np.round(norm.cdf(he * v, loc=5, scale=4/6), 3) * (v >= 2)

        # aggregate damage ratings
        dr1 = frag1
        dr2 = frag2_1f + frag2_2f
        dr3 = frag3_1f + frag3_2f
        dr4 = frag4
        dr5 = frag5_1f + frag5_2f
        dr6 = frag6_1f + frag6_2f
        dr7 = frag7
        dr8 = frag8
        
        return (dr1, dr2, dr3, dr4, dr5, dr6, dr7, dr8)
    
    def compute_damage(self, 
        he:float, 
        v:float, 
        d:float, 
        s:float, 
        q:bool
        ) -> dict:

        """Compute damages from a flood event.
       
        Args:
            he: external water height (m).
            v: water velocity (m/s).
            d: flood duration (hrs).
            s: sediment concentration.
            q: presence of pollutants (binary).

        """

        h = self.waterLevel(he)
        he = np.minimum(he, self.NF * self.IH * 1.05)    
        dr = self.fragility(he, v, d)

        nf = np.repeat(self.NF, len(he))

        # C1: Pumping (€/m3)
        C1 = self.costs["pumping"] * (he >= 0) * (
            self.IA * max(-self.GL, 0) +
            self.BA * (-self.BL - np.minimum(0.3, ((self.GL > 0) & (self.GL < 0.3)) * (0.3 - self.GL)))
            ) * (1 - 0.2 * (self.BT == 3))

        # C2: Waste disposal (€/m3)
        C2 = self.costs["disposal"] * \
            s * (1 + q * 0.4) * (
            self.IA * h +
            self.BA * self.BH
            ) * (1 - 0.2 * (self.BT == 3))

        # C3: Cleaning (€/m2)
        C3 = self.costs["cleaning"] * \
            (1 + q * 0.4) * (
            self.IA * np.minimum(nf, np.ceil(h / self.IH)) + self.IP * h +
            self.BA + self.BP * self.BH
            ) * (1 - 0.2 * (self.BT == 3))

        # C4: Dehumidification (€/m3)
        C4 = self.costs["dehumidification"] * dr[0] * (
            self.IA * self.IH * np.minimum(nf, np.ceil(h / self.IH)) * (he > 0) +
            self.BA * self.BH
            ) * (1 - 0.2 * (self.BT == 3))

        # 2. Removal
        # ==========
        # Screed removal
        R1 = self.costs["screedremoval"] * self.IA * (
            (self.FL > 1) * dr[0] * np.minimum(nf, dr[1])
        ) * (1 - 0.2 * (self.BT == 3))

        # Pavement removal
        R2 = self.costs["parquetremoval"] * (self.FL > 1) * \
            dr[0] * np.minimum(nf, dr[1]) * self.IA * (1 - 0.2 * (self.BT == 3))

        # Baseboard removal
        R3 = self.costs["baseboardremoval"] * \
            dr[0] * np.minimum(nf, np.ceil((h - 0.05) / self.IH)) * self.IP * (1 - 0.2 * (self.BT == 3))

        # Partitions removal
        R4 = self.costs["partitionsremoval"] * dr[0] * \
            (1 + ((self.BS == 1) * 0.20)) * 0.5 * self.IP * self.IH * np.minimum(nf, dr[2]) * (1 - 0.2 * (self.BT == 3))

        # Plasterboard removal
        R5 = self.costs["plasterboardremoval"] * \
            self.IA * 0.2 * np.minimum(nf, np.ceil((h - (self.IH - 0.5)) / self.IH)) * (self.FL > 1) * (1 - 0.2 * (self.BT == 3))

        # External plaster removal
        R6 = self.costs["extplasterremoval"] * \
            max(int(q), int(self.LM <= 1), dr[0], dr[3]) * self.EP * (he + 1.0) * (1 - 0.2 * (self.BT == 3))

        # Internal plaster removal
        R7 = self.costs["intplasterremoval"] * \
            max(int(q), int(self.LM <= 1), dr[0]) * (self.IP * (h + 1.0) + self.BP * self.BH) * (1 - 0.2 * (self.BT == 3))

        # Doors removal
        R8 = self.costs["doorsremoval"] * \
            max(dr[3], dr[0]) * (np.minimum(nf, dr[4]) * 0.12 * self.IA + 0.03 * self.BA) * (1 - 0.2 * (self.BT == 3))

        # Windows removal
        R9 = self.costs["windowsremoval"] * \
            max(dr[6], dr[0]) * (np.minimum(nf, dr[5]) * 0.12 * self.IA) * (1 - 0.2 * (self.BT == 3))

        # Boiler removal
        R10 = self.costs["boilerremoval"] * self.IA * (
            (self.PD == 1) * (int(self.BA > 0) + (int(self.BA == 0) * (h > 1.6))) +
            (self.PD == 2) * np.minimum(nf, np.ceil((h - 1.6) / self.IH))
        ) * (1 - 0.2 * (self.BT == 3))


        # 3. Non-Structural
        # =================
        # Partitions replacement
        N1 = self.costs["partitionsreplace"] * dr[0] * \
            (1 + ((self.BS == 1) * 0.20)) * 0.5 * self.IP * self.IH * np.minimum(nf, dr[2]) * (1 - 0.2 * (self.BT == 3))

        # Screed replacement
        N2 = self.costs["screedreplace"] * self.IA * (
            (self.FL > 1) * dr[0] * np.minimum(nf, dr[1])
        ) * (1 - 0.2 * (self.BT == 3))

        # Plasterboard replacement
        N3 = self.costs["plasterboardreplace"] * \
            self.IA * 0.2 * np.minimum(nf, np.ceil((h - (self.IH - 0.5)) / self.IH)) * (self.FL > 1) * (1 - 0.2 * (self.BT == 3))


        # 4. Structural
        # =============
        S1 = self.costs["soilconsolidation"] * dr[7] * self.FA * self.NF * self.IH * (0.01 + ((self.BS == 1) * 0.01)) * (1 - 0.2 * (self.BT == 3))

        S2 = self.costs["localrepair"] * (self.BS == 2) * dr8 * self.EP * 0.5 * he * (1 + s) * (1 - 0.2 * (self.BT == 3))

        S3 = self.costs["pillarretrofitting"] * (self.BS == 1) * dr[7] * 0.15 * self.EP * he * (1 - 0.2 * (self.BT == 3))


        # 5. Finishing
        # ============
        # External plaster replacement
        F1 = self.costs["extplasterreplace"] * self.FL * \
            max(int(q), (self.LM <= 1), dr[0], dr[3]) * self.EP * (he + 1.0) * (1 - 0.2 * (self.BT == 3))

        # Internal plaster replacement
        F2 = self.costs["intplasterreplace"] * self.FL * \
            max(int(q), (self.LM <= 1), dr1) * (self.IP * (h + 1.0) + self.BP * self.BH) * (1 - 0.2 * (self.BT == 3))

        # Painting
        F3 = self.costs["extpainting"] * np.minimum(nf, np.ceil(he / self.IH)) * self.IH * self.EP * self.FL * (1 - 0.2 * (self.BT == 3))
        
        F4 = self.costs["intpainting"] * (
            np.minimum(nf, np.ceil(h / self.IH)) * self.IH * self.IP + self.BP * self.BH * (1 if (self.FL > 1 and self.BT == 1) else 0)
        ) * self.FL * (1 - 0.2 * (self.BT == 3))

        # Pavement replacement 
        F5 = self.costs["parquetreplace"] * (self.FL > 1) * dr[0] * np.minimum(nf, dr[1]) * self.IA * (1 - 0.2 * (self.BT == 3))

        # Baseboard replacement
        F6 = self.costs["baseboardreplace"] * dr[0] * np.minimum(nf, np.ceil((h - 0.05) / self.IH)) * self.IP * (1 - 0.2 * (self.BT == 3))

        # 6. Windows and doors
        # ====================
        # Doors replacement
        W1 = self.costs["doorsreplace"] * max(dr[3], dr[0]) * (np.minimum(nf, dr[4]) * 0.12 * self.IA + 0.03 * self.BA) * (1 + (self.FL > 1)) * (1 - 0.2 * (self.BT == 3))

        # Windows replacement
        W2 = self.costs["windowsreplace"] * max(dr[6], dr[0]) * (np.minimum(nf, dr[5]) * 0.12 * self.IA) * (1 + (self.FL > 1)) * (1 - 0.2 * (self.BT == 3))


        # 7. Building systems
        # ===================
        # Boiler replacement
        P1 = self.costs["boilerreplace"] * self.IA * (
            (self.PD == 1) * ((self.BA > 0) + ((self.BA == 0) * (h > 1.6))) +
            (self.PD == 2) * np.minimum(nf, np.ceil((h - 1.6) / self.IH))
        ) * (1 + 0.25 * ((self.BT == 1) ^ (self.BT == 2)))

        # Radiator painting
        P2 = self.costs["radiatorpaint"] * (self.PT == 1) * np.minimum(nf, np.ceil((h - 0.2) / self.IH)) * (self.IA / 20) * (1 - 0.2 * (self.BT == 3))

        # Underfloor heating replacement
        P3 = self.costs["underfloorheatingreplace"] * self.IA * (self.PT == 2) * ((self.FL > 1) * dr[0] * np.minimum(nf, dr[1])) * (1 - 0.2 * (self.BT == 3))

        # Electrical system replacement
        P4 = self.costs["electricalsystreplace"] * self.IA * (
            np.minimum(nf, np.ceil((h - 0.2) / self.IH)) * 0.4 +
            np.minimum(nf, np.ceil((h - 1.1) / self.IH)) * 0.3 +
            np.minimum(nf, np.ceil((h - 1.5) / self.IH)) * 0.3
        ) * (1 + (self.FL > 1)) * (1 - 0.2 * (self.BT == 3))

        # Plumbing system replacement
        P5 = up["plumbingsystreplace"] * self.IA * (
            ((s > 0.10) or q) * (
                np.minimum(nf, np.ceil((h - 0.15) / self.IH)) * 0.1 +
                np.minimum(nf, np.ceil((h - 0.4) / self.IH)) * 0.2 +
                np.minimum(nf, np.ceil((h - 0.9) / self.IH)) * 0.2
            )
        ) * (1 + (self.FL > 1)) * (1 - 0.2 * (self.BT == 3))

        
        # Sum the damage components
        dmgCleanUp      = C1 + C2 + C3
        dmgRemoval      = R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10
        dmgNonStructural = N1 + N2 + N3
        dmgStructural   = S1 + S2 + S3
        dmgFinishing    = F1 + F2 + F3 + F4 + F5 + W1 + W2
        dmgSystems      = P1 + P2 + P3 + P4 + P5

        absDamage = dmgCleanUp + dmgRemoval + dmgNonStructural + dmgStructural + dmgFinishing + dmgSystems

        relDamage = absDamage / self.RVN

        groupDamage = {
            "dmgCleanUp": dmgCleanUp,
            "dmgRemoval": dmgRemoval,
            "dmgNonStructural": dmgNonStructural,
            "dmgStructural": dmgStructural,
            "dmgFinishing": dmgFinishing,
            "dmgSystems": dmgSystems
        }

        componentDamage = {
            "C1": C1, "C2": C2, "C3": C3,
            "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5,
            "R6": R6, "R7": R7, "R8": R8, "R9": R9, "R10": R10,
            "N1": N1, "N2": N2, "N3": N3,
            "S1": S1, "S2": S2, "S3": S3,
            "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5,
            "W1": W1, "W2": W2,
            "P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5
        }

        result = {
            "waterLevel": h,
            "absDamage": absDamage,
            "relDamage": relDamage,
            "groupDamage": groupDamage,
            "componentDamage": componentDamage
        }

        return result

    
