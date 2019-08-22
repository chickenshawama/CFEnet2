import pickle
import os
root='E:/work/ssd/CFEnet/ssd300_120000/val'
filename=os.listdir(root)
classes=['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic light', 'traffic sign', 'train']
for cls in classes:
    cachefile=os.path.join(root,cls + '_pr.pkl')
    with open(cachefile, 'rb') as f:
        recs = pickle.load(f)
        print(cls)
        print(recs['ap'])