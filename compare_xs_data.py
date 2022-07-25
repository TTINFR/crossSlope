import pandas as pd
import numpy as np
import math
#Y:\AT_2021\Deliveries\Delivery_20220228\MAIN\xslope_new_calibration
#Y:\AT_2020\Deliveries\Delivery_20201130\MAIN\xslope
#Y:\AT_2019\Deliveries\Delivery_20191130\MAIN\xslope
df1 = pd.read_csv(r"Y:\Users\claire.martin1\cross_slope_new\crossSlope\method_20m\csvs\02116s_a_vux.csv")
df2 = pd.read_csv(r"Y:\AT_2020\Deliveries\Delivery_20201130\MAIN\xslope\02116_xs.csv")

num_wrong_types = 0
differences_crown = []
differences_super = []
types_wrong = []
""" NEG = False

if NEG:
    df2 = df2[df2["Lane_Code"] != "R1"]
    df2 = df2.set_index(pd.Index(range(0,len(df2["Crown_Slope"])))) """

""" df2 = df2[df2["Roadway_Code"] == "L1"]
df2 = df2[df2["Lane_Code"] == "L1"]
df2 = df2[df2["From_km_DMI"] < 40]
df2 = df2.set_index(pd.Index(range(0,len(df2["Crown_Slope"])))) """

df2 = df2[df2["Lane_Code"] != "R1"]
df2 = df2[df2["Lane_Code"] != "R2"]
df2 = df2.set_index(pd.Index(range(0,len(df2["Crown_Slope"]))))

#print(df2["Lane_Code"])
for i in range(len(df1["Crown_Slope"])):
    slope1 = df1["Crown_Slope"][i]
    slope2 = df2["Crown_Slope"][i]
    slope1 = abs(float(slope1))
    try:
        slope2 = abs(float(slope2.split("%")[0]))
    except:
        slope2 = abs(float(slope2))
    if np.isnan(slope1) and np.isnan(slope2):
        # skip this one
        continue
    elif (np.isnan(slope1) and not np.isnan(slope2)) or (np.isnan(slope2) and not np.isnan(slope1)):
        # got the types wrong
        num_wrong_types += 1
        types_wrong.append(i)
        continue
    else:
        differences_crown.append(abs(slope1 - slope2))

for i in range(len(df1["Super_Elevation"])):
    slope1 = df1["Super_Elevation"][i]
    slope2 = df2["Super_Elevation"][i]
    slope1 = abs(float(slope1))
    try:
        slope2 = abs(float(slope2.split("%")[0]))
    except:
        slope2 = abs(float(slope2))
    if np.isnan(slope1) and np.isnan(slope2):
        continue
    elif i not in types_wrong:
        differences_super.append(abs(slope1 - slope2))

print("Total types wrong: %d"%num_wrong_types)
print("Types wrong for indexes: ", types_wrong)
print("Avg difference for crowns: %.3f"%np.mean(differences_crown))
print("Avg difference for super: %.3f"%np.mean(differences_super))