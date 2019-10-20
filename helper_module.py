import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

import cv2 ## openCV
import os
import numpy as np
import matplotlib.pyplot as plt
import operator

from IPython.display import Markdown, display
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

original_path = os.getcwd()
## OpenCV haarcascade model for "face detection"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
## for input images
img_width = 550
img_height = 750
## for cropped face
img_col = 96
img_row = 96
## hyperparams
box_size_factor = 10 # bigger value allows smaller bounding box
face_recog_thresh = 0.70

## `embed_image` to embed processed face images into 128d vectors
def embed_image(face_img,model):
    '''
    embed the RGB cropped face (input) into 128d vector
    use with `detect_face()`
    '''
    img = cv2.resize(face_img, (img_col,img_row)).astype('float32')
    img /= 255.0
    img = np.expand_dims(img,axis=0)
    embedding = model.predict_on_batch(img)
    return embedding

## `detect_face` to detect frontal faces in *gray* (higher accuracy than doing it in color) 
def detect_face(img,fc=face_cascade,flag='db',plot=False):
    '''
    Receive BGR format as an input and return coordinate(s) of detected face(s)
    
    default: flag = 'db' --> assume only one face is present in the image and return only 1 face
    flag = 'new' --> if to embed new images (possibly multiple faces)
    '''
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height,width = img_grey.shape
    faces_raw = fc.detectMultiScale(img_grey) # higher accuracy for faces with black glasses
    faces = []
    
    # get rid of errorneous small boxes
    for face in faces_raw:
        if face[2] > (min(height,width)/box_size_factor):
            faces.append(face)
            
    if flag == 'db':
        face_box = [0,0,0,0]
        for (x,y,w,h) in faces:
            if w > face_box[2]:
                face_box = [x,y,w,h] # IGNOTE ALL OTHER FALSY FACE BOXES for database embedding
        (x,y,w,h) = face_box
        faces = [face_box]       
    if flag == 'new':
        faces = faces
        
    if plot:
        num_col = 5
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_color_for_crop = img_color.copy()
        for (plot_x,plot_y,plot_w,plot_h) in faces:   
            img_color = cv2.rectangle(img_color, (plot_x,plot_y), (plot_x+plot_w,plot_y+plot_h), (255,0,0), 8)

        plt.title('full image',fontdict={'fontsize':15,'fontweight':'bold'})
        plt.imshow(img_color)
        plt.axis('off')
        if len(faces) == 1:
            (plot_x,plot_y,plot_w,plot_h) = faces[0]
            fig,ax=plt.subplots(1,1,figsize=(3,3))
            cropped = img_color_for_crop[plot_y:plot_y+plot_h,plot_x:plot_x+plot_w]
            ax.imshow(cropped)
            ax.axis('off')
            fig.suptitle('Cropped face image to be embedded',fontsize=15,fontweight='bold')  
        elif len(faces)<=num_col:
            fig,axes=plt.subplots(1,len(faces),figsize=(3*len(faces),3))
            for ax,(plot_x,plot_y,plot_w,plot_h) in zip(axes.flatten(),faces):
                cropped = img_color_for_crop[plot_y:plot_y+plot_h,plot_x:plot_x+plot_w]
                ax.imshow(cropped)
                ax.axis('off')
            fig.suptitle('Cropped face image to be embedded (not ordered)',fontsize=15,fontweight='bold')
        else:
            fig, axes = plt.subplots(int(np.ceil(len(faces)/num_col)),num_col,figsize=(15,3*int(np.ceil(len(faces)/num_col))))
            fig.suptitle('Cropped face image to be embedded (not ordered)',fontsize=15,fontweight='bold')

            for ax,(plot_x,plot_y,plot_w,plot_h) in zip(axes.flatten(),faces):
                cropped = img_color_for_crop[plot_y:plot_y+plot_h,plot_x:plot_x+plot_w]
                ax.imshow(cropped)
                ax.axis('off')
            if not len(faces)==len(axes.flatten()):
                for i in axes.flatten()[len(faces)-len(axes.flatten()):]:
                    i.set_visible(False)
    return faces

## `database_face_embedding` to process and embed images in the database
def database_face_embedding(model):
    '''
    embed the images in the database - folder name 'image_database' required
    output = {'name':embedding,...}
    '''
    database_embeddings = {}
    os.chdir(os.path.join(os.getcwd(),'image_database'))
    for img_file in os.listdir():
        name = img_file.split('.')[0]
        image_file = cv2.imread(img_file)
        image_file = cv2.resize(image_file,(img_width,img_height), interpolation = cv2.INTER_AREA)
        faces = detect_face(image_file)
        (x, y, w, h) = faces[0]
        image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
        cropped = image_file[y:y+h,x:x+w]
        database_embeddings[name] = embed_image(cropped, model)
    os.chdir(original_path)
    return database_embeddings

## `identify_singe_face` to identify person given a single face image
def identify_singe_face(new_face,database_embeddings,model,face_recog_thresh,verbose=None): 
    '''
    receive one new RGB face as an input
    return name_label of that face as one of the registered members
    '''    
    new_face_embedding = embed_image(new_face,model)
    name_label = ''
    result = {}
    min_dist = 100
    for (registered_name,registered_embedding) in database_embeddings.items():
        euc_dist = np.linalg.norm(new_face_embedding-registered_embedding)
        euc_dist = round(euc_dist,3)
        result[registered_name] = euc_dist

        if euc_dist < min_dist:
            min_dist = euc_dist
            name = registered_name
    if min_dist < face_recog_thresh:
        if verbose:
        	printmd('@@@ this is '+'**{}**'.format(name.upper())+'! @@@\n',color='red')
        	print('Distance from:')
        	for i in sorted(result.items(),key=operator.itemgetter(1)):
        		if i[0] == name:
        			printmd('**{}**'.format(i),color='red')
        		else:
        			print(i)
        	print('')
        name_label = name
        return name_label
    else:
    	if verbose:
    		print('@@@ not registered! @@@\n')
    		print('Distance from:')
    		for i in sorted(result.items(),key=operator.itemgetter(1)):
    			print(i)
    		print('')
    	name_label = 'n/a'
    	return name_label


## `recog_face` to recognize multiple faces in a single frame (image)
def recog_face(img,database_embeddings,model,face_recog_thresh=0.7,fc=face_cascade,verbose=None):
    '''
    receive BGR image as an input
    return image with overlayed bounding boxes and names of the registered members
    ''' 
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detect_face(img,flag='new')
        
    for (x, y, w, h) in faces:
        cropped = img_color[y:y+h,x:x+w]
        if verbose:
            name = identify_singe_face(cropped,database_embeddings,model,face_recog_thresh,verbose=True)
        if not verbose:
            name = identify_singe_face(cropped,database_embeddings,model,face_recog_thresh)
        
        text = '{}'.format(name)
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN,3,5)[0]
        text_offset_x = x-3
        text_offset_y = y
        box_coords = ((text_offset_x, text_offset_y+10), (text_offset_x+text_width,text_offset_y-text_height-10))
        
        if name != 'n/a':
            img_color = cv2.rectangle(img_color, (x, y), (x+w, y+h), (255,0,0), 8)
            img_color = cv2.rectangle(img_color, box_coords[0], box_coords[1], (255,0,0), cv2.FILLED)
            img_color = cv2.putText(img_color,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),4)
        if name == 'n/a':
        	img_color = cv2.rectangle(img_color, (x, y), (x+w, y+h), (0,0,255), 8)
        	img_color = cv2.rectangle(img_color, box_coords[0], box_coords[1], (0,0,255), cv2.FILLED)
        	img_color = cv2.putText(img_color,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),4)

    plt.figure(figsize=(8,8))
    plt.imshow(img_color)
    plt.axis('off')   

## load and preprocess the image
def load_test_img(file):
	img = cv2.imread(os.path.join('test',file))
	img = cv2.resize(img,(img_width,img_height),interpolation=cv2.INTER_AREA)
	return img

