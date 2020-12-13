# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:14:38 2020

@author: asega
"""
import numpy as np
import imageio
import matplotlib.pyplot as plt
import DLT 


# %% Picture from side
img1=np.array(imageio.imread('D:\Travail\ENSAM-01\BME\Medical Imaging\Project\P1010891.jpg'))
plt.axis('auto')
plt.imshow(img1)

xyz_side = [[0.2012,0.2298,0.0994], [0.2213,0.2153,0.4698], [0.3367,0.1758,0.9343], [0.2933,0.0505,1.6459], [0.2360,0.1954,1.3776], [0.2238,0.2763,1.2192], [0.2207,0.4,0.958],
       [0.2236,0.4392,0.8676], [0.30399, 0.2764, 0.047], [0.3042,0.1227,1.5221], [0.4032,0.1136,1.5570], [0.3601, 0.0413, 1.3137], [0.1210,0.1611,0.0754], [0.296,0.2112,0.0959], [0.7591, 0.3996, 0.0225], [0.8063,-0.2002, 0.0212], [-0.1141, 0.4357, 0.01952], [-0.0959, -0.3269, 0.0181]]
uv_side = [[891, 1590], [876, 1267], [775, 858], [802, 257], [864, 466], [880, 601], [895, 840], [890, 927], [802.5,1647.2], [804, 351], [715, 325], [751, 544], [956, 1588], [807, 1583], [380,1714], [420, 1539], [1201,1730], [1073, 1512]]

# %% Picture from face

img2=np.array(imageio.imread('D:\Travail\ENSAM-01\BME\Medical Imaging\Project\P1010890.jpg'))
plt.axis('auto')
plt.imshow(img2)

xyz_face = [[0.2360, 0.1954, 1.3776], [0.2460, -0.1171, 1.3667], [0.3785, 0.0459, 1.1740], [0.3601, 0.0413, 1.3137], [0.2155, -0.1949, 1.1866], [0.2238, 0.2763, 1.21924], [0.3212, -0.0696, 0.9301],
        [0.3367, 0.1759, 0.9343], [0.2434, -0.1242, 0.5041], [0.2213, 0.2153, 0.4698], [0.2521, -0.0085, 0.4595], [0.2414, 0.1247, 0.4658], [0.3125, -0.1220, 0.0974], [0.2960, 0.2112, 0.0959],[0.8063, -0.2002, 0.0212], [0.7591, 0.3996, 0.0225], [-0.1141, 0.4357, 0.01952], [-0.0959, -0.3269, 0.0181]]

uv_face = [[2043, 816], [1371, 833], [1718, 1226], [1713, 921], [1202, 1227], [2212, 1165.7], [1456, 1774], [2006, 1774], [1340, 2677], [2061, 2748], [1591, 2774], [1864, 2762], [1338, 3559], [2049, 3560],[1058,4147],[2604,4111],[2413,3475],[999,3476]]

# %% Open the 3D obj
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

model= open('D:\Travail\ENSAM-01\BME\Medical Imaging\Project\Model.txt','r')
a=[]
for line in model:
    ligne=line.split(' ')
    a.append(ligne)
    #print (ligne)
a.pop(0)
points=[]
for i in range(len(a)): #Extraire les lignes avec 'v'
    if a[i][0]=='v':
        points.append([float(a[i][1]),float(a[i][2]),float(a[i][3]),1])
points=np.array(points)

plt.plot(points[:,0],points[:,1],points[:,2],"o",label='model reconstruit')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')
plt.axis('auto')
plt.legend()
plt.show()
points=np.transpose(points)
# %% Traitement des données
angle=232*np.pi/180
dx=0.3
dy=0
dz=0.15
Matrice_rotation=np.array([[np.cos(angle),0,np.sin(angle),dx],[0,1,0,dy],[-np.sin(angle),0,np.cos(angle),dz],[0,0,0,1]])
points=Matrice_rotation.dot(points)
store=np.copy(points[1,:])
points[1,:]=np.copy(points[2,:])
points[2,:]=store
# %% Plan sol

XYZ_face=np.concatenate((np.transpose(np.array(xyz_face)),np.ones((1,18))))
XYZ_side=np.concatenate((np.transpose(np.array(xyz_side)),np.ones((1,18))))
sol=np.array([XYZ_face[:,-1],XYZ_face[:,-2],XYZ_face[:,-3],XYZ_face[:,-4]])
inv_sol=np.linalg.inv(sol)
[a,b,c,d]=np.transpose(inv_sol.dot(np.transpose([1,1,1,1])))
normale=(1/np.sqrt(a**2+b**2+c**2))*np.transpose([a,b,c])

# Proj_x=normale[0]
# Proj_y=normale[1]
# Proj_z=normale[2]
# theta=np.arccos(Proj_x/np.linalg.norm(normale))
# phi=np.arccos(Proj_y/np.linalg.norm(normale))
# psi=np.arccos(Proj_z/np.linalg.norm(normale))
# M_x=np.array([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
# M_y=np.array([[np.cos(phi),0,np.sin(phi),0],[0,1,0,0],[-np.sin(phi),0,np.cos(phi),0],[0,0,0,1]])
# M_z=np.array([[np.cos(psi),-np.sin(psi),0,0],[np.sin(psi),np.cos(psi),0,0],[0,0,1,0],[0,0,0,1]])
# M_r=M_x*M_y*M_z
# points=M_r.dot(points)

# %% Affichage du modèle tourné
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(points[0,:],points[1,:],points[2,:],"o",label='model tourné')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')
plt.axis('auto')
plt.legend()
plt.show()
# %% Projection dans le plan
Matrice_face=DLT.DLTcalib(3,xyz_face,uv_face)[0]
Matrice_side=DLT.DLTcalib(3,xyz_side,uv_side)[0]
uv_face_model_t=np.dot(Matrice_face,points)
uv_side_model_t=np.dot(Matrice_side,points)
# %% Extraction des coordonées
uv_face_model=[]
uv_side_model=[]
for i in range(len(uv_face_model_t)):
    uv_face_model.append([uv_face_model_t[i][0],uv_face_model_t[i][1]])
    uv_side_model.append([uv_side_model_t[i][0],uv_side_model_t[i][1]])
uv_face_model=np.array(uv_face_model)
uv_face_model=uv_face_model/uv_face_model[2,:]
uv_side_model=np.array(uv_side_model)
uv_side_model=uv_side_model/uv_side_model[2,:]
# %% Affichage

fig=plt.figure()
plt.imshow(img2)
plt.plot(uv_face_model_t[0,:],uv_face_model_t[1,:],'o')

fig=plt.figure()
plt.imshow(img1)
plt.plot(uv_side_model_t[0,:],uv_side_model_t[1,:],'o')
#%% comparaison DLT()

UV_face=np.concatenate((np.transpose(np.array(uv_face)),np.ones((1,18))))
inv_face=np.linalg.pinv(Matrice_face)
UV_face_calc=Matrice_face.dot(XYZ_face)
UV_face_calc=UV_face_calc/UV_face_calc[2,:]
Dif_UV_face=UV_face-UV_face_calc
XYZ_face_calc=inv_face.dot(UV_face)
# XYZ_face_calc=XYZ_face_calc/XYZ_face_calc[3,:]
Dif_XYZ_face=XYZ_face-XYZ_face_calc
Err_XYZ_face=[]
Err_UV_face=[]
for i in range(len(Dif_XYZ_face[0,:])):
    Err_XYZ_face.append(Dif_XYZ_face[0][i]**2+Dif_XYZ_face[1][i]**2+Dif_XYZ_face[2][i]**2)
    Err_UV_face.append(Dif_UV_face[0][i]**2+Dif_UV_face[1][i]**2)
RMS_XYZ_face=np.sqrt(np.mean(Err_XYZ_face))
RMS_UV_face=np.sqrt(np.mean(Err_UV_face))



UV_side=np.concatenate((np.transpose(np.array(uv_side)),np.ones((1,18))))
inv_side=np.linalg.pinv(Matrice_side)
UV_side_calc=Matrice_side.dot(XYZ_side)
UV_side_calc=UV_side_calc/UV_side_calc[2,:]
Dif_UV_side=UV_side-UV_side_calc
XYZ_side_calc=inv_side.dot(UV_side)
# XYZ_side_calc=XYZ_side_calc/XYZ_side_calc[3,:]
Dif_XYZ_side=XYZ_side-XYZ_side_calc
Err_XYZ_side=[]
Err_UV_side=[]
for i in range(len(Dif_XYZ_side[0,:])):
    Err_XYZ_side.append(Dif_XYZ_side[0][i]**2+Dif_XYZ_side[1][i]**2+Dif_XYZ_side[2][i]**2)
    Err_UV_side.append(Dif_UV_side[0][i]**2+Dif_UV_side[1][i]**2)
RMS_XYZ_side=np.sqrt(np.mean(Err_XYZ_side))
RMS_UV_side=np.sqrt(np.mean(Err_UV_side))
# %% Affichage backprojection
fig=plt.figure()


plt.imshow(img2)
plt.plot(UV_face[0,:],UV_face[1,:],'o',label='marqueurs pointés')
#plt.plot(UV_face_calc[0,:],UV_face_calc[1,:],'o',label='marqueurs calculés')
plt.legend()
plt.show()

fig=plt.figure()

plt.imshow(img1)
plt.plot(UV_side[0,:],UV_side[1,:],'o',label='marqueurs pointés')
#plt.plot(UV_side_calc[0,:],UV_side_calc[1,:],'o',label='marqueurs calculés')
plt.legend()
plt.show()

# %% Affichage 3D

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(XYZ_face[0,:],XYZ_face[1,:],XYZ_face[2,:],label='marqueurs pointés')
ax.scatter(XYZ_face_calc[0,:],XYZ_face_calc[1,:],XYZ_face_calc[2,:],label='marqueurs calculés')
ax.legend()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(XYZ_side[0,:],XYZ_side[1,:],XYZ_side[2,:],label='marqueurs pointés')
ax.scatter(XYZ_side_calc[0,:],XYZ_side_calc[1,:],XYZ_side_calc[2,:],label='marqueurs calculés')
ax.legend()