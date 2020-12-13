# Body-shape-from-2-photos

This is a school project from the BME master program in Paris. It is following the Medical imaging and geometrical modeling class. It was created by Alexandre SEGAIN, Xiao MA and Anton KANIEWSKI. We based our work on several other projects and we modified the code to match with our purpose. The purpose of the code is to create a 3D body shape model front photos. At least two photos, one from the front and another from the side.
If you use this code, don't forget to respect their licence please.

Enjoy !

------ Required libraries ------

Python 3.6 
numpy 1.15.2 
scikit-image 0.14.0 
python-opencv PIL 5.2.0 
PyTorch 0.4.0 
torchvision 0.2.1 glob

------ Tutorial ------

1.Please put the front and side pictures to be tested in the "input" folder and modify them to "front.jpg" and "side.jpg".     path:./input

2.open ’ main.py’  with python  

3.Input the height and run this code.

4.You can find the resulting model in the ‘output’ folder. path: ./output

5.You can also find the original Silhouette cutting picture in this path:   ./ Silhouette

6. Go to ‘./output’, right-click the model, open the model with ‘Pint 3D’software. Then select the ‘3D shapes’ button at the top of the Pint 3D interface, and then click the ‘select’ button in the lower column. Left-click on the model, a rectangular frame will appear, and then right-click to select "Horizontal Flip". Click Save as ’. glb’ file. the 3d body shape model has been established.


------ References ------

U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection Xuebin Qin, Zichen Zhang,Chenyang Huang, Masood Dehghan, Osmar R. Zaiane and Martin Jagersand. https://github.com/NathanUA/U-2-Net#u2-net-going-deeper-with-nested-u-structure-for-salient-object-detection

Concise and Effective Network for 3D Human Modeling from Orthogonal Silhouettes Bin Liua, Xiuping Liua, Zhixin Yangb,Charlie C.L. Wangc https://github.com/liubindlut/SilhouettesbasedHSE#readme

------ Citations ------

@InProceedings{Qin_2020_PR, title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection}, author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin}, journal = {Pattern Recognition}, volume = {106}, pages = {107404}, year = {2020} }
