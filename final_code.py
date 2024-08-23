import fitz
from PIL import Image
import io
import os
import re
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
import pandas as pd
import torch
import shutil

def extract_images_from_pdf(pdf_path, output_dir, extracted_text_file):
    """Extract images from a PDF and save them with corresponding text."""
    pdf_document = fitz.open(pdf_path)
    image_data_list = []

    with open(extracted_text_file, 'w') as text_file:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)

            for img in page.get_images(full=True):
                if 250 <= img[2] <= 2000 and 250 <= img[3] <= 2000:
                    xref = img[0]
                    image_bbox = page.get_image_bbox(img)
                    x0, y0, x1, y1 = image_bbox

                    text_above = page.get_textbox((x0, y0 - 50, x1, y0))
                    text_below = page.get_textbox((x0, y1, x1, y1 + 50))
                    text = (text_above + " " + text_below).strip()
                    cleaned_text = re.sub("[^a-zA-Z0-9 ]", '', text)
                    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()[:250]

                    # Preprocess text to match classification categories
                    preprocessed_text = preprocess(cleaned_text)
                    sanitized_filename = re.sub(r'\s+', '_', preprocessed_text)  
                    sanitized_filename = re.sub(r'[^\w\s-]', '', sanitized_filename)  
                    sanitized_filename = sanitized_filename.strip().lower()

                    print(f"Extracted and preprocessed text for image on page {page_number + 1}: {sanitized_filename}")

                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    if not image_ext:
                        image_ext = 'png'  # default png

                    image = Image.open(io.BytesIO(image_bytes))
                    image = image.convert("RGB")
                    image_filename = f"page_{page_number + 1}_image_{sanitized_filename}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)

                    print(f"Saved image to {image_path}")

                    image_data = {
                        "image_path": image_path,
                        "text": preprocessed_text
                    }
                    image_data_list.append(image_data)

                    # Write the preprocessed text to the extracted text file
                    text_file.write(preprocessed_text + '\n')

    json_path = os.path.join(output_dir, "extracted_data.json")
    with open(json_path, "w") as f:
        json.dump(image_data_list, f, indent=4)

    print(f"Extracted data saved to {json_path}")

def preprocess(text):
    """Preprocess text by removing stop words and punctuation, and lemmatizing."""
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def preprocess_file(input_file_path, output_file_path):
    """Read the original file, process the text, and write back the processed text."""
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = [preprocess(line.strip()) for line in lines]

    with open(output_file_path, 'w') as file:
        for processed_line in processed_lines:
            file.write(processed_line + '\n')

    for processed_line in processed_lines[:10]:
        print(processed_line)

def cluster_headings(input_file_path, output_file_path, num_clusters=10):
    """Cluster headings from the processed text file using KMeans."""
    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"File not found: {input_file_path}")

    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        preprocessed_lines = [preprocess(line.strip()) for line in lines]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_lines)

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    df = pd.DataFrame({'heading': lines, 'cluster': clusters})
    df.to_csv(output_file_path, index=False)

def classify_headings(input_file_path, output_file_path):
    """Classify headings using a pre-trained model for zero-shot classification."""
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=device)

    categories = [
        "similarity factor", "types of images", "location map", "geo map", "structural map",
        "seismic section", "logmotive", "well construction diagram", "geotechnical order",
        "remote sensing image", "contour maps", "drilling plot", "cartography", "casing plot"
    ]

    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("Error: The file 'processed_names.txt' was not found.")
        return

    results = []
    for text in lines:
        text = text.strip()
        if not text:
            print("Skipping empty line.")
            continue

        try:
            result = classifier(text, candidate_labels=categories)
            results.append({
                'heading': text,
                'predicted_category': result['labels'][0],
                'score': result['scores'][0]
            })
        except Exception as e:
            print(f"Error classifying text '{text}': {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file_path, index=False)
        print(f"Classification complete. Results saved to '{output_file_path}'.")
    else:
        print("No valid results to save.")
        
def organize_images_by_category(extracted_data_json, classified_csv, output_dir):
    """Organize images into folders based on their classified categories."""
    
    with open(extracted_data_json, 'r') as f:
        image_data_list = json.load(f)
    
    
    df = pd.read_csv(classified_csv)

   
    df['heading_lower'] = df['heading'].str.lower()
    
    
    categories = df['predicted_category'].unique()

    
    for category in categories:
        category_dir = os.path.join(output_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
    
    
    unclassified_dir = os.path.join(output_dir, 'unclassified')
    if not os.path.exists(unclassified_dir):
        os.makedirs(unclassified_dir)

   
    for image_data in image_data_list:
        image_path = image_data['image_path']
        heading_text = image_data['text'].lower()  # Convert to lowercase

        print(f"Processing image: {image_path}")
        print(f"Extracted text: {heading_text}")

        # Find the classification row with case-insensitive matching
        classified_row = df[df['heading_lower'].str.contains(heading_text, na=False)]

        if not classified_row.empty:
            category = classified_row.iloc[0]['predicted_category']
            print(f"Classified as: {category}")
        else:
            category = 'unclassified'
            print("No classification found, using 'unclassified'")

        # Move image
        category_dir = os.path.join(output_dir, category)
        destination_path = os.path.join(category_dir, os.path.basename(image_path))
        
        if not os.path.exists(destination_path):
            shutil.move(image_path, destination_path)
            print(f"Moved {image_path} to {destination_path}")
        else:
            print(f"Skipped moving {image_path} as {destination_path} already exists")


if __name__ == "__main__":
    pdf_path = "well_student.pdf"
    output_dir = "new5"
    extracted_text_file = "extracted_names.txt"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the spaCy model
    nlp = spacy.load('en_core_web_md')
    
    extract_images_from_pdf(pdf_path, output_dir, extracted_text_file)

    input_file_path = extracted_text_file
    output_file_path = 'processed_names.txt'
    preprocess_file(input_file_path, output_file_path)

    cluster_headings(output_file_path, 'clustered_headings.csv')
    classify_headings(output_file_path, 'classified_headings.csv')
    
    organize_images_by_category(os.path.join(output_dir, 'extracted_data.json'), 'classified_headings.csv', output_dir)
