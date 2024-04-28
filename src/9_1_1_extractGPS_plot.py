import pandas as pd
import utm
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

data = pd.read_csv('DJIFlightRecord.csv', encoding='windows-1252')

coords_data = data.iloc[:, [12, 13, 15]].copy()
coords_data.columns = ['OSD.latitude', 'OSD.longitude', 'OSD.altitude']

record_state_data = data['CAMERA_INFO.recordState'].copy()
coords_data_rec = pd.DataFrame(record_state_data, columns=['CAMERA_INFO.recordState'])

for index, row in coords_data.iterrows():
    latitude = row['OSD.latitude']
    longitude = row['OSD.longitude']
    # print(latitude, longitude)

    #convert to UTM
    utm_coords = utm.from_latlon(latitude, longitude)
    # print(utm_coords)

    coords_data.loc[index, 'UTM.easting'] = utm_coords[0]
    coords_data.loc[index, 'UTM.northing'] = utm_coords[1]
    coords_data.loc[index, 'UTM.zone_number'] = utm_coords[2]
    coords_data.loc[index, 'UTM.zone_letter'] = utm_coords[3]

    if coords_data_rec.at[index, 'CAMERA_INFO.recordState'] == 'Starting':
        coords_data_rec.loc[index, 'UTM.easting'] = utm_coords[0]
        coords_data_rec.loc[index, 'UTM.northing'] = utm_coords[1]
        coords_data_rec.loc[index, 'OSD.altitude'] = row['OSD.altitude']

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(coords_data['UTM.easting'], coords_data['UTM.northing'], coords_data['OSD.altitude'])
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.set_xlabel('UTM Easting')
ax.set_ylabel('UTM Northing')
ax.set_zlabel('Altitude')

fig_rec = plt.figure()
ax_rec = plt.axes(projection='3d')
ax_rec.plot3D(coords_data_rec['UTM.easting'], coords_data_rec['UTM.northing'], coords_data_rec['OSD.altitude'])
ax_rec.axes.get_xaxis().set_ticks([])
ax_rec.axes.get_yaxis().set_ticks([])
ax_rec.set_xlabel('UTM Easting')
ax_rec.set_ylabel('UTM Northing')
ax_rec.set_zlabel('Altitude')

plt.show()