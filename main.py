import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset".
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width =4 ,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab',
                    'Apple___Black_rot', 
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Blueberry___healthy', 
                    'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 
                    'Peach___Bacterial_spot',
                    'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 
                    'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 
                    'Potato___Late_blight', 
                    'Potato___healthy', 
                    'Raspberry___healthy', 
                    'Soybean___healthy',
                    'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch',
                    'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight',
                    'Tomato___Late_blight', 
                    'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        st.success("Model is Predicting it's a {} ".format(class_name[result_index]))
    
        
        
    
        class_names = ['Apple scab, caused by the fungus Venturia inaequalis, affects apple trees worldwide, leading to reduced fruit quality and yield. Control measures include cultural practices like removing fallen leaves, pruning to improve air circulation, and planting resistant varieties. Fungicides can be effective, especially when applied preventively. Organic options include sulfur and copper-based fungicides. Timing is crucial, with applications starting at bud break and continuing through the growing season. Regular monitoring and early intervention are key to managing apple scab. Combining different control methods in an integrated approach can enhance effectiveness and reduce the reliance on chemicals, promoting sustainable orchard management.',
                    'Apple black rot, caused by the fungus Botryosphaeria obtusa, is a serious disease affecting apple trees. To control it, practice good orchard hygiene by removing and destroying infected leaves, fruit, and prunings. Prune trees to improve air circulation and sunlight penetration. Apply fungicides preventively starting at petal fall, with repeat applications according to the product label. Organic options include copper-based fungicides and biological controls. Implementing a regular spray schedule is crucial for managing black rot. Monitor your orchard closely and remove any infected material promptly. These practices, when combined, can effectively reduce the impact of black rot on apple trees.', 
                    'Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae, affecting apple trees and Eastern red cedar trees. Control measures include planting resistant apple varieties and removing nearby Eastern red cedars, which serve as the alternate host. Fungicides can be applied preventively, starting at the pink bud stage and continuing at intervals specified on the product label. Organic options include sulfur and copper-based fungicides. Pruning to improve air circulation and removing infected plant material can also help manage cedar apple rust. Implementing a comprehensive management plan that combines these strategies can effectively reduce the impact of this disease.',
                    'There is no disease on the Apple leaf',
                    'There is no disease on the Blueberry leaf', 
                    'Powdery mildew on cherry trees, caused by the fungus Podosphaera clandestina, can be managed through several methods. Planting resistant varieties can reduce the risk of infection. Cultural practices such as pruning to improve air circulation and reducing humidity around the trees can also help. Fungicides, including sulfur and potassium bicarbonate, can be applied preventively or at the first sign of infection. Organic options like neem oil are also effective. Regular monitoring of trees and prompt treatment can prevent the spread of powdery mildew. Implementing a combination of these strategies can help control and manage powdery mildew on cherry trees.', 
                    'There is no disease on the Cherry_(including_sour) leaf', 
                    'Cercospora leaf spot and gray leaf spot, caused by the fungi Cercospora zeae-maydis and Cercospora zeina, respectively, are common diseases of corn (maize). To manage these diseases, plant resistant varieties when possible. Rotate crops to reduce disease pressure, as these pathogens can survive on crop residue. Use fungicides preventively, starting at early stages of crop development and continuing at intervals recommended on the product label. Cultural practices such as planting in well-drained soil and avoiding excessive nitrogen fertilization can also help. Regular scouting and monitoring for symptoms are crucial for early detection and effective management of these diseases.', 
                    'Common rust, caused by the fungus Puccinia sorghi, is a common disease of corn (maize). To manage common rust, plant resistant corn varieties when possible. Practice crop rotation to reduce disease pressure. Fungicides can be used preventively, starting at the early stages of crop development and continuing at intervals recommended on the product label. Cultural practices such as proper spacing between plants and adequate soil drainage can also help manage common rust. Regular scouting and monitoring for symptoms are crucial for early detection and effective management of common rust in corn.', 
                    'Northern leaf blight, caused by the fungus Exserohilum turcicum, is a damaging disease of corn (maize). To manage Northern leaf blight, plant resistant corn varieties if available. Practice crop rotation to reduce disease pressure, as the fungus can survive on crop residue. Fungicides can be applied preventively, starting at the early stages of crop development and continuing at intervals specified on the product label. Cultural practices such as planting in well-drained soil and avoiding excessive nitrogen fertilization can also help reduce the severity of Northern leaf blight. Regular scouting and monitoring for symptoms are essential for early detection and effective management of this disease.',
                    'There is no disease on the Corn_(maize) leaf', 
                    'Black rot, caused by the fungus Guignardia bidwellii, is a common and destructive disease of grapes. To manage black rot, practice good vineyard sanitation by removing and destroying infected plant material. Prune vines to improve air circulation and sunlight penetration. Apply fungicides preventively, starting at bud break and continuing at regular intervals according to the product label. Organic options include sulfur and copper-based fungicides. Cultural practices such as proper spacing between vines and canopy management can also help reduce the risk of black rot. Regular monitoring and early intervention are key to managing this disease and preserving grape yield and quality.', 
                    'Esca, also known as black measles, is a complex disease of grapevines with various causal agents. Control measures include planting disease-free nursery stock, managing vine stress, and improving vineyard hygiene by removing infected wood. Fungicides can be used preventively, although their effectiveness is limited. Trunk surgery and protectant applications can be employed, but these methods are not always successful. Research into biological control agents and resistant cultivars is ongoing. Proper vineyard management practices, including balanced nutrition and canopy management, can help reduce esca incidence. Regular monitoring and prompt removal of infected vines are crucial for managing esca in grapevines.',
                    'Leaf blight, caused by the fungus Isariopsis griseola, is a common disease of grapevines. To manage leaf blight, practice good vineyard sanitation by removing and destroying infected leaves. Apply fungicides preventively, starting at the first sign of disease and continuing at regular intervals according to the product label. Organic options include sulfur and copper-based fungicides. Cultural practices such as proper spacing between vines and canopy management can also help reduce the risk of leaf blight. Regular monitoring and early intervention are key to managing this disease and preserving grape yield and quality.', 
                    'There is no disease on the Grape leaf', 
                    'Huanglongbing (HLB), also known as citrus greening, is a devastating disease of citrus trees caused by the bacterium Candidatus Liberibacter asiaticus. There is no cure for HLB, making management challenging. Control measures include planting disease-free nursery stock, controlling the Asian citrus psyllid vector, and removing and destroying infected trees. Nutritional programs and stress management can help prolong tree productivity. Research into resistant citrus varieties and bactericidal treatments is ongoing but has not yet resulted in a practical solution. Early detection and rapid response are critical to minimizing the spread of HLB.', 
                    'Bacterial spot, caused by the bacterium Xanthomonas arboricola pv. pruni, is a common disease of peach trees. To manage bacterial spot, practice good orchard sanitation by removing and destroying infected plant material. Apply copper-based fungicides or bactericides preventively, starting at bud swell and continuing at regular intervals according to the product label. Pruning to improve air circulation and sunlight penetration can also help reduce the severity of bacterial spot. Avoid overhead irrigation to minimize the spread of bacteria. Regular monitoring and early intervention are crucial for managing bacterial spot and preserving peach yield and quality.',
                    'There is no disease on the Peach leaf', 
                    'Bacterial spot, caused by the bacterium Xanthomonas campestris pv. vesicatoria, is a common disease of bell peppers. To manage bacterial spot, practice good garden hygiene by removing and destroying infected plant material. Apply copper-based fungicides or bactericides preventively, starting at the first sign of disease and continuing at regular intervals according to the product label. Avoid overhead irrigation to minimize the spread of bacteria. Proper spacing between plants and adequate ventilation can also help reduce the severity of bacterial spot. Regular monitoring and early intervention are key to managing this disease and preserving bell pepper yield and quality.', 
                    'There is no disease on the Bell Pepper leaf', 
                    'Early blight, caused by the fungus Alternaria solani, is a common disease of potato plants. To manage early blight, practice crop rotation and avoid planting potatoes in the same area for consecutive years. Remove and destroy infected plant debris. Apply fungicides preventively, starting when plants are young and continuing at intervals recommended on the product label. Cultural practices such as adequate spacing between plants and proper irrigation management can also help reduce the severity of early blight. Regular monitoring for symptoms and early intervention are crucial for managing this disease and preserving potato yield and quality.', 
                    'Late blight, caused by the oomycete Phytophthora infestans, is a devastating disease of potatoes. To manage late blight, practice good crop rotation and avoid planting potatoes in areas where the disease has been present in previous seasons. Remove and destroy infected plant material to reduce the spread of the pathogen. Apply fungicides preventively, starting before the disease appears and continuing at intervals specified on the product label. Cultural practices such as proper spacing between plants and adequate ventilation can also help reduce the severity of late blight. Regular monitoring and early intervention are crucial for managing this disease and preserving potato yield and quality.', 
                    'There is no disease on the Potato leaf', 
                    'There is no disease on the Raspberry leaf', 
                    'There is no disease on the soybean leaf',
                    'Powdery mildew, caused by the fungus Podosphaera xanthii, is a common disease of squash plants. To manage powdery mildew, practice good garden hygiene by removing and destroying infected plant material. Ensure proper spacing between plants for good air circulation and sunlight penetration. Apply fungicides preventively, starting when plants are young and continuing at intervals recommended on the product label. Organic options include sulfur and potassium bicarbonate-based fungicides. Neem oil can also be effective. Cultural practices such as watering at the base of plants and avoiding overhead irrigation can help reduce the severity of powdery mildew. Regular monitoring and early intervention are key to managing this disease', 
                    'Leaf scorch, caused by the fungus Diplocarpon earlianum, is a common disease of strawberry plants. To manage leaf scorch, practice good garden hygiene by removing and destroying infected plant material. Ensure proper spacing between plants for good air circulation. Apply fungicides preventively, starting at the first sign of disease and continuing at intervals specified on the product label. Organic options include sulfur and copper-based fungicides. Cultural practices such as mulching to reduce soil splashing and watering at the base of plants can also help reduce the severity of leaf scorch. Regular monitoring and early intervention are key to managing this disease.',
                    'There is no disease on the Strawberry leaf', 
                    'Copper fungicides are the most commonly recommended treatment for bacterial leaf spot. Use copper fungicide as a preventive measure after you’ve planted your seeds but before you’ve moved the plants into their permanent homes. You can use copper fungicide spray before or after a rain, but don’t treat with copper fungicide while it is raining. If you’re seeing signs of bacterial leaf spot, spray with copper fungicide for a seven- to 10-day period, then spray again for one week after plants are moved into the field. Perform maintenance treatments every 10 days in dry weather and every five to seven days in rainy weather.', 
                    'Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic..',
                    'Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic.', 
                    ' Use drip irrigation and avoid watering foliage. Use a stake, strings, or prune the plant to keep it upstanding and increase airflow in and around it. Remove and destroy (burn) all plants debris after the harvest.', 
                    'Consider organic fungicide options: Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease. Begin spraying as soon as the first symptoms appear and follow the label directions for continued management',
                    'For control, use selective products whenever possible. Selective products which have worked well in the field include: # bifenazate (Acramite): Group UN, a long residual nerve poison.# abamectin (Agri-Mek): Group 6, derived from a soil bacterium. # spirotetramat (Movento): Group 23, mainly affects immature stages.# spiromesifen (Oberon 2SC): Group 23, mainly affects immature stages', 
                    'Many fungicides are registered to control of target spot on tomatoes. Growers should consult regional disease management guides for recommended products. Products containing chlorothalonil, mancozeb, and copper oxychloride have been shown to provide good control of target spot in research trials', 
                    'Inspect plants for whitefly infestations two times per week. If whiteflies are beginning to appear, spray with azadirachtin (Neem), pyrethrin or insecticidal soap. For more effective control, it is recommended that at least two of the above insecticides be rotated at each spraying.',
                    'There are no cures for viral diseases such as mosaic once a plant is infected. As a result, every effort should be made to prevent the disease from entering your garden.',
                    'There is no disease on the Tomato leaf']
        st.success("Treatment :-  {} ".format(class_names[model_prediction(test_image)]))