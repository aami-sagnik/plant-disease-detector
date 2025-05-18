import re
from pathlib import Path

def get_is_healthy_plant_disease(category):
    category = category.split(' ')[0]
    is_healthy = "healthy" in category.lower()
    fragments = re.findall(r'[A-Z][^A-Z]*', category)
    plant = fragments[0].split("_")[0].lower()
    plant = re.sub(r'[^a-zA-Z]', ' ', plant)
    plant = plant.strip().lower()
    if not is_healthy:
        disease = "".join(fragments[1:])
        disease = re.sub(r'[^a-zA-Z()]', ' ', disease)
        disease = disease.strip().lower()
    else:
        disease = None
    return { 
        "is_healthy": is_healthy, 
        "plant": plant, 
        "disease": disease 
        }


def get_all_images_by_category(plant, disease, image_list):
    if disease is None:
        disease = "healthy"
        
    images_by_plant = [ image_path for image_path in image_list if plant in image_path.parent.stem.lower() ]
    selected_images = []
    
    for image_path in images_by_plant:
        fragments = disease.split(" ")
        found = True
        for fragment in fragments:
            if fragment not in image_path.parent.stem.lower():
                found = False
                break
        if found:
            selected_images.append(image_path)
    
    return selected_images

