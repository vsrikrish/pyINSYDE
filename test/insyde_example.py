from src.pyinsyde.insyde import BuildingProperties, Building

prop = {
'FA': 100,
'IA': 90,
'BA': 50,
'EP': 40,
'IH': 3.5,
'BH': 3.2,
'GL': 0.1,
'NF': 1,
'BT': 1,
'BS': 2,
'PD': 1,
'PT': 1,
'FL': 1.2,
'YY': 1994,
'LM': 1.1
}

bd = BuildingProperties(**prop)
b = Building(bd)

he = 3  # water depth (m)
v = 0.5   # velocity (m/s)
s = 0.05  # sediment concentration (-)
d = 24    # flood duration (h)
q = 1     # water quality (presence of pollutants) 

h = b.waterLevel(he)
res = b.compute_damage(he, v, d, s, q)

print(round(res["absDamage"], 0) == 65849.0)