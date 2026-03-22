"""
Medical Knowledge Base
=======================
Curated medical knowledge entries covering radiology, pathology, and anatomy.
Used for RAG-based knowledge injection into the VQA model prompt.

Knowledge sources modeled after:
- UMLS (Unified Medical Language System) concepts
- SNOMED CT (Systematized Nomenclature of Medicine - Clinical Terms)
- Radiology reference facts
- PubMed abstract summaries
"""

from typing import List, Dict


def get_medical_knowledge_base() -> List[Dict[str, str]]:
    """
    Returns a curated list of medical knowledge entries.
    
    Each entry contains:
        - concept: Medical concept or term
        - definition: Detailed medical definition
        - category: Classification (anatomy, pathology, radiology, etc.)
        - source: Knowledge source attribution
    
    Returns:
        List of medical knowledge dictionaries.
    """
    knowledge = [
        # ============================================================
        # CHEST / THORACIC RADIOLOGY
        # ============================================================
        {
            "concept": "Cardiomegaly",
            "definition": "Cardiomegaly is an enlarged heart, typically defined as a cardiothoracic ratio greater than 0.5 on a posteroanterior chest radiograph. It may indicate congestive heart failure, valvular disease, cardiomyopathy, or pericardial effusion. The cardiac silhouette appears wider than half the thoracic diameter.",
            "category": "cardiology",
            "source": "UMLS:C0018800"
        },
        {
            "concept": "Pleural Effusion",
            "definition": "Pleural effusion is the abnormal accumulation of fluid in the pleural space between the visceral and parietal pleura. On chest X-ray, it appears as blunting of the costophrenic angle (meniscus sign). Common causes include heart failure, pneumonia, malignancy, and pulmonary embolism. Large effusions may cause mediastinal shift.",
            "category": "pulmonology",
            "source": "UMLS:C0032227"
        },
        {
            "concept": "Pneumothorax",
            "definition": "Pneumothorax is the presence of air in the pleural space, causing partial or complete lung collapse. On chest X-ray, it appears as a visible pleural line with absent lung markings beyond it. Types include spontaneous, traumatic, and tension pneumothorax. Tension pneumothorax is a medical emergency.",
            "category": "pulmonology",
            "source": "UMLS:C0032326"
        },
        {
            "concept": "Pulmonary Consolidation",
            "definition": "Consolidation refers to the replacement of air in the alveoli with fluid, pus, blood, or cells. On imaging, it appears as an opacified area with air bronchograms. Common causes include pneumonia (bacterial, viral, fungal), pulmonary hemorrhage, and organizing pneumonia. Lobar consolidation suggests bacterial pneumonia.",
            "category": "pulmonology",
            "source": "UMLS:C0521530"
        },
        {
            "concept": "Atelectasis",
            "definition": "Atelectasis is the partial or complete collapse of lung tissue. It can be obstructive (endobronchial lesion), compressive (pleural effusion), or passive. On chest X-ray, signs include increased opacity, volume loss, mediastinal shift toward the affected side, and elevation of the hemidiaphragm.",
            "category": "pulmonology",
            "source": "UMLS:C0004144"
        },
        {
            "concept": "Pulmonary Edema",
            "definition": "Pulmonary edema is the accumulation of fluid in the lung parenchyma and alveoli. Cardiogenic pulmonary edema shows bilateral perihilar opacities (butterfly pattern), Kerley B lines, peribronchial cuffing, and pleural effusions. Non-cardiogenic (ARDS) shows diffuse bilateral opacities without cardiomegaly.",
            "category": "pulmonology",
            "source": "UMLS:C0034063"
        },
        {
            "concept": "Pneumonia",
            "definition": "Pneumonia is infection of the lung parenchyma. Bacterial pneumonia typically shows lobar consolidation with air bronchograms. Viral pneumonia shows bilateral interstitial infiltrates. Aspiration pneumonia involves dependent lung segments. Key imaging findings include ground-glass opacities, consolidation, and tree-in-bud pattern.",
            "category": "infectious_disease",
            "source": "UMLS:C0032285"
        },
        {
            "concept": "Lung Mass / Pulmonary Nodule",
            "definition": "A pulmonary nodule is a round opacity less than 3 cm in diameter. A pulmonary mass is greater than 3 cm. Differential diagnosis includes primary lung cancer, metastasis, granuloma, hamartoma. Malignant features include spiculated margins, size > 3cm, upper lobe location, and growth on serial imaging.",
            "category": "oncology",
            "source": "UMLS:C0034072"
        },
        {
            "concept": "Mediastinal Widening",
            "definition": "Mediastinal widening on chest X-ray (>8 cm on PA view) may indicate aortic dissection, aortic aneurysm, lymphadenopathy, mediastinal mass, or traumatic aortic injury. It is a critical finding that requires further evaluation with CT angiography.",
            "category": "cardiology",
            "source": "UMLS:C0240318"
        },
        {
            "concept": "Chest X-ray Normal Anatomy",
            "definition": "Normal chest X-ray shows clear lung fields bilaterally, normal cardiac silhouette (CTR < 0.5), sharp costophrenic angles, normal mediastinal contour, visible trachea in midline, and normal bony thorax. The right hemidiaphragm is normally slightly higher than the left due to the liver.",
            "category": "anatomy",
            "source": "SNOMED:399208008"
        },
        {
            "concept": "Hilar Lymphadenopathy",
            "definition": "Hilar lymphadenopathy refers to enlargement of lymph nodes at the lung hila. Bilateral hilar lymphadenopathy is characteristic of sarcoidosis, lymphoma, and metastatic disease. Unilateral hilar lymphadenopathy may indicate lung cancer, tuberculosis, or lymphoma.",
            "category": "pulmonology",
            "source": "UMLS:C0520743"
        },
        {
            "concept": "Interstitial Lung Disease",
            "definition": "Interstitial lung disease (ILD) encompasses a group of disorders affecting the lung interstitium. On imaging, findings include reticular opacities, ground-glass opacities, honeycombing, and traction bronchiectasis. Distribution pattern helps narrow differential: upper lobe (silicosis, sarcoidosis), lower lobe (IPF, asbestosis).",
            "category": "pulmonology",
            "source": "UMLS:C0206062"
        },
        {
            "concept": "Tuberculosis",
            "definition": "Tuberculosis (TB) on chest X-ray may show upper lobe infiltrates, cavitation, miliary pattern (diffuse tiny nodules), hilar lymphadenopathy, and pleural effusion. Primary TB in children shows lymphadenopathy and Ghon complex. Reactivation TB typically involves apical and posterior segments of upper lobes.",
            "category": "infectious_disease",
            "source": "UMLS:C0041296"
        },
        {
            "concept": "COPD / Emphysema",
            "definition": "Chronic obstructive pulmonary disease shows hyperinflated lungs, flattened diaphragms, increased retrosternal airspace, and barrel chest on lateral view. Emphysema specifically shows decreased vascular markings and bullae. The heart may appear elongated and narrow (tubular heart).",
            "category": "pulmonology",
            "source": "UMLS:C0024115"
        },
        
        # ============================================================
        # ABDOMINAL IMAGING
        # ============================================================
        {
            "concept": "Hepatomegaly",
            "definition": "Hepatomegaly is enlargement of the liver beyond its normal size. On imaging, the liver extends below the right costal margin. Common causes include fatty liver disease, hepatitis, cirrhosis, congestive heart failure, and malignancy. Normal liver span is approximately 12-15 cm in the midclavicular line.",
            "category": "gastroenterology",
            "source": "UMLS:C0019209"
        },
        {
            "concept": "Ascites",
            "definition": "Ascites is the pathological accumulation of fluid in the peritoneal cavity. On imaging, it appears as fluid-density material surrounding bowel loops. Common causes include cirrhosis (most common), heart failure, malignancy, and nephrotic syndrome. CT shows dependent fluid and may demonstrate peritoneal enhancement in malignant ascites.",
            "category": "gastroenterology",
            "source": "UMLS:C0003962"
        },
        {
            "concept": "Bowel Obstruction",
            "definition": "Bowel obstruction is blockage of the intestinal lumen. Small bowel obstruction shows dilated loops (>3 cm) with air-fluid levels and decompressed distal bowel. Large bowel obstruction shows colonic dilation (>6 cm, cecum >9 cm). Causes include adhesions, hernias, tumors, and volvulus.",
            "category": "gastroenterology",
            "source": "UMLS:C0021843"
        },
        
        # ============================================================
        # NEUROIMAGING
        # ============================================================
        {
            "concept": "Intracranial Hemorrhage",
            "definition": "Intracranial hemorrhage types include epidural (lens-shaped, arterial), subdural (crescent-shaped, venous), subarachnoid (in sulci and cisterns), intraparenchymal (within brain tissue), and intraventricular. Acute blood appears hyperdense on CT. Hemorrhage is a medical emergency requiring urgent evaluation.",
            "category": "neurology",
            "source": "UMLS:C0151699"
        },
        {
            "concept": "Ischemic Stroke",
            "definition": "Ischemic stroke results from arterial occlusion causing brain infarction. Early CT signs include loss of gray-white differentiation, sulcal effacement, hyperdense vessel sign, and insular ribbon sign. MRI DWI shows restricted diffusion within minutes. Territory distribution helps identify the occluded vessel.",
            "category": "neurology",
            "source": "UMLS:C0948008"
        },
        {
            "concept": "Brain Tumor",
            "definition": "Brain tumors on imaging may show enhancing mass lesion, surrounding edema, mass effect, and midline shift. Primary brain tumors include glioblastoma (ring-enhancing), meningioma (dural-based, homogeneously enhancing), and acoustic neuroma (CP angle mass). Metastases are typically multiple, at gray-white junction.",
            "category": "neuro-oncology",
            "source": "UMLS:C0006118"
        },
        
        # ============================================================
        # MUSCULOSKELETAL
        # ============================================================
        {
            "concept": "Fracture",
            "definition": "A fracture is a break in bone continuity. Fracture types include transverse, oblique, spiral, comminuted, and avulsion. Important features to describe: location, type, displacement, angulation, and associated soft tissue injury. Stress fractures may only be visible on MRI initially.",
            "category": "orthopedics",
            "source": "UMLS:C0016658"
        },
        {
            "concept": "Osteoarthritis",
            "definition": "Osteoarthritis shows joint space narrowing, osteophytes, subchondral sclerosis, and subchondral cysts on radiographs. It is the most common form of arthritis, typically affecting weight-bearing joints. Unlike rheumatoid arthritis, it is asymmetric and does not show periarticular osteoporosis.",
            "category": "rheumatology",
            "source": "UMLS:C0029408"
        },
        
        # ============================================================
        # PATHOLOGY
        # ============================================================
        {
            "concept": "Malignant Neoplasm",
            "definition": "Malignant neoplasms (cancers) are characterized by uncontrolled cell growth, invasion, and metastasis. Histological features include cellular atypia, increased mitotic activity, nuclear pleomorphism, and loss of differentiation. Imaging features include irregular margins, heterogeneous enhancement, and lymphadenopathy.",
            "category": "oncology",
            "source": "UMLS:C0006826"
        },
        {
            "concept": "Inflammation",
            "definition": "Inflammation is the body's immune response to injury or infection. Cardinal signs include rubor (redness), calor (heat), tumor (swelling), dolor (pain), and functio laesa (loss of function). Acute inflammation shows neutrophilic infiltration; chronic inflammation shows lymphocytic and macrophage infiltration.",
            "category": "pathology",
            "source": "UMLS:C0021368"
        },
        {
            "concept": "Fibrosis",
            "definition": "Fibrosis is the formation of excess fibrous connective tissue in an organ. In the lungs, it appears as reticular opacities, honeycombing, and traction bronchiectasis. In the liver, it leads to cirrhosis. Fibrosis represents irreversible tissue remodeling and scarring from chronic injury.",
            "category": "pathology",
            "source": "UMLS:C0016059"
        },
        {
            "concept": "Necrosis",
            "definition": "Necrosis is pathological cell death due to injury. Types include coagulative (ischemia), liquefactive (brain, abscess), caseous (tuberculosis), fat necrosis, and fibrinoid necrosis. On imaging, necrosis often appears as non-enhancing areas within a lesion.",
            "category": "pathology",
            "source": "UMLS:C0027540"
        },
        {
            "concept": "Edema",
            "definition": "Edema is the accumulation of excess fluid in tissues. Pulmonary edema shows fluid in lung parenchyma. Cerebral edema shows brain swelling with sulcal effacement. Peripheral edema affects subcutaneous tissues. Causes include heart failure, renal failure, liver disease, and lymphatic obstruction.",
            "category": "pathology",
            "source": "UMLS:C0013604"
        },
        
        # ============================================================
        # IMAGING MODALITIES
        # ============================================================
        {
            "concept": "X-ray (Radiography)",
            "definition": "X-ray imaging uses electromagnetic radiation to produce 2D projection images. Dense structures (bone) appear white, air appears black, and soft tissues appear in gray. Chest X-ray is the most common radiological examination. PA (posteroanterior) view is standard for chest imaging.",
            "category": "radiology",
            "source": "SNOMED:363680008"
        },
        {
            "concept": "CT (Computed Tomography)",
            "definition": "CT scanning uses rotating X-ray beams to create cross-sectional images. It provides superior soft tissue contrast compared to plain radiography. CT uses Hounsfield units (HU) for density measurement: air (-1000), fat (-100), water (0), soft tissue (+40), bone (+1000). Contrast enhancement helps characterize lesions.",
            "category": "radiology",
            "source": "SNOMED:77477000"
        },
        {
            "concept": "MRI (Magnetic Resonance Imaging)",
            "definition": "MRI uses magnetic fields and radiofrequency pulses to create detailed soft tissue images. T1-weighted images show fat as bright and fluid as dark. T2-weighted images show fluid as bright. DWI (diffusion-weighted imaging) detects acute ischemia. MRI has no ionizing radiation.",
            "category": "radiology",
            "source": "SNOMED:113091000"
        },
        {
            "concept": "Ultrasound",
            "definition": "Ultrasound uses high-frequency sound waves to create real-time images. It is excellent for evaluating fluid-filled structures, the heart (echocardiography), obstetrics, and guiding procedures. Doppler ultrasound evaluates blood flow. No ionizing radiation makes it safe in pregnancy.",
            "category": "radiology",
            "source": "SNOMED:16310003"
        },
        
        # ============================================================
        # ANATOMY
        # ============================================================
        {
            "concept": "Heart Anatomy",
            "definition": "The heart has four chambers: right atrium, right ventricle, left atrium, left ventricle. The right side receives deoxygenated blood and pumps to lungs via pulmonary arteries. The left side receives oxygenated blood and pumps to body via aorta. Key valves: tricuspid, pulmonic, mitral, aortic.",
            "category": "anatomy",
            "source": "SNOMED:80891009"
        },
        {
            "concept": "Lung Anatomy",
            "definition": "The right lung has three lobes (upper, middle, lower) and the left lung has two lobes (upper, lower) with the lingula. The trachea bifurcates at the carina into right and left main bronchi. The right main bronchus is wider, shorter, and more vertical, making it more prone to aspiration.",
            "category": "anatomy",
            "source": "SNOMED:39607008"
        },
        {
            "concept": "Mediastinum",
            "definition": "The mediastinum is the central compartment of the thorax between the lungs. It contains the heart, great vessels, trachea, esophagus, thoracic duct, and lymph nodes. Divided into anterior, middle, and posterior compartments. Anterior mediastinal masses: thymoma, teratoma, terrible lymphoma, thyroid.",
            "category": "anatomy",
            "source": "SNOMED:72410000"
        },
        {
            "concept": "Diaphragm",
            "definition": "The diaphragm is the primary muscle of respiration separating thorax from abdomen. The right hemidiaphragm is normally slightly higher than the left. Diaphragmatic elevation may indicate phrenic nerve palsy, subdiaphragmatic process, or hepatomegaly. Diaphragmatic hernia allows abdominal contents into thorax.",
            "category": "anatomy",
            "source": "SNOMED:5798000"
        },
        
        # ============================================================
        # CLINICAL CONCEPTS
        # ============================================================
        {
            "concept": "Differential Diagnosis",
            "definition": "Differential diagnosis is the process of distinguishing between diseases with similar presentations. In medical imaging, it involves considering the most likely diagnoses based on imaging patterns, patient demographics, clinical history, and location of findings. The most common diagnosis should be listed first.",
            "category": "clinical",
            "source": "UMLS:C0011906"
        },
        {
            "concept": "Radiological Report Structure",
            "definition": "A standard radiology report includes: clinical indication, technique/protocol, comparison with prior studies, findings (organized by organ system or anatomical region), and impression (summary of key findings with differential diagnosis). Critical findings require direct verbal communication.",
            "category": "radiology",
            "source": "SNOMED:imaging_report"
        },
        {
            "concept": "BIRADS Classification",
            "definition": "BI-RADS (Breast Imaging Reporting and Data System) categories: 0-incomplete, 1-negative, 2-benign, 3-probably benign (<2% malignancy risk, short-term follow-up), 4-suspicious (4a: 2-10%, 4b: 10-50%, 4c: 50-95%), 5-highly suggestive of malignancy (>95%), 6-known malignancy.",
            "category": "radiology",
            "source": "UMLS:C1709157"
        },
        {
            "concept": "Contrast Enhancement Patterns",
            "definition": "Contrast enhancement patterns help characterize lesions. Homogeneous enhancement suggests benign lesion. Ring/rim enhancement suggests abscess or necrotic tumor. Heterogeneous enhancement suggests malignancy. Non-enhancing areas indicate necrosis or cyst. Arterial phase hyperenhancement with washout suggests hepatocellular carcinoma.",
            "category": "radiology",
            "source": "SNOMED:contrast_patterns"
        },
        {
            "concept": "Ground Glass Opacity",
            "definition": "Ground glass opacity (GGO) is a hazy increase in lung density that does not obscure underlying bronchial or vascular structures. It is less dense than consolidation. Causes include pneumonia (viral, PCP), pulmonary hemorrhage, pulmonary edema, and early fibrosis. COVID-19 characteristically shows peripheral bilateral GGOs.",
            "category": "pulmonology",
            "source": "UMLS:C3544344"
        },
        {
            "concept": "Air Bronchogram",
            "definition": "Air bronchogram is the visualization of air-filled bronchi within an area of lung opacification (consolidation or atelectasis). It indicates that the airway is patent and the surrounding alveoli are filled with fluid or cells. Most commonly seen in pneumonia and pulmonary edema.",
            "category": "radiology",
            "source": "UMLS:C0238508"
        },
        {
            "concept": "Costophrenic Angle",
            "definition": "The costophrenic angles are the junctions of the diaphragm and the chest wall on a chest radiograph. Normally they are sharp and acute. Blunting of the costophrenic angle is the earliest sign of pleural effusion on an upright chest X-ray, typically indicating at least 200-300 mL of fluid.",
            "category": "anatomy",
            "source": "SNOMED:costophrenic"
        },
        {
            "concept": "Cardiothoracic Ratio",
            "definition": "The cardiothoracic ratio (CTR) is the ratio of the maximum transverse cardiac diameter to the maximum transverse thoracic diameter on a PA chest X-ray. Normal CTR is less than 0.5 (50%). A CTR greater than 0.5 indicates cardiomegaly. AP films may falsely elevate the CTR.",
            "category": "cardiology",
            "source": "UMLS:C0428699"
        },
        {
            "concept": "Reticular Pattern",
            "definition": "Reticular pattern on chest imaging consists of innumerable interlacing line shadows suggesting a net-like pattern. It indicates thickening of the interstitial structures. Causes include interstitial pulmonary fibrosis, lymphangitic carcinomatosis, and pulmonary edema. Fine reticular pattern suggests early fibrosis.",
            "category": "radiology",
            "source": "UMLS:reticular_pattern"
        },
        {
            "concept": "Tracheal Deviation",
            "definition": "Tracheal deviation from midline on chest X-ray is an important finding. Deviation toward a lesion suggests atelectasis or prior surgery. Deviation away from a lesion suggests tension pneumothorax, large pleural effusion, or mediastinal mass. Tracheal deviation is best evaluated on PA chest radiograph.",
            "category": "radiology",
            "source": "UMLS:C0577936"
        },
        {
            "concept": "Silhouette Sign",
            "definition": "The silhouette sign occurs when a structure of water density (heart, aorta, diaphragm) is in contact with a pathological water-density process (consolidation, mass), causing loss of the normal silhouette. Loss of the right heart border suggests right middle lobe pathology. Loss of the left heart border suggests lingula pathology.",
            "category": "radiology",
            "source": "UMLS:silhouette_sign"
        },
    ]
    
    return knowledge


def get_knowledge_categories() -> List[str]:
    """Get all unique categories in the knowledge base."""
    kb = get_medical_knowledge_base()
    return sorted(list(set(entry["category"] for entry in kb)))
