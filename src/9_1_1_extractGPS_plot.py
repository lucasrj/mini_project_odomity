import pandas as pd
import utm
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

data = pd.read_csv('DJIFlightRecord.csv')

coords_data = data.iloc[:, 13:16].copy()
coords_data.columns = ['OSD.latitude', 'OSD.longitude', 'OSD.height']

for index, row in coords_data.iterrows():
    latitude = row['OSD.latitude']
    longitude = row['OSD.longitude']

    #convert to UTM
    utm_coords = utm.from_latlon(latitude, longitude)

    coords_data.loc[index, 'UTM.easting'] = utm_coords[0]
    coords_data.loc[index, 'UTM.northing'] = utm_coords[1]
    coords_data.loc[index, 'UTM.zone_number'] = utm_coords[2]
    coords_data.loc[index, 'UTM.zone_letter'] = utm_coords[3]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(coords_data['UTM.easting'], coords_data['UTM.northing'], coords_data['OSD.height'])
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.set_xlabel('UTM Easting')
ax.set_ylabel('UTM Northing')
ax.set_zlabel('Height')
plt.show()