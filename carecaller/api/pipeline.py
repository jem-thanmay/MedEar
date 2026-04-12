import sys
import os
import re
import json
import torch
import whisper
import audioop
import numpy as np
import soundfile as sf
import static_ffmpeg

sys.path = [p for p in sys.path if '3.12' not in p]
static_ffmpeg.add_paths()

from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/whisper-medical")

print("Loading MedEar model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

base_model = whisper.load_model("small")
print(f"✅ MedEar loaded on {device}")

PHONETIC_CORRECTIONS = {
    # lisinopril
    "lecinopril": "lisinopril",
    "lecinoprol": "lisinopril",
    "leesinopril": "lisinopril",
    "lysineopril": "lisinopril",
    "lysineaberyl": "lisinopril",
    "licinopril": "lisinopril",
    "listen or fill": "lisinopril",
    "listen or phil": "lisinopril",
    "lissen or fill": "lisinopril",
    "lisenorfill": "lisinopril",
    "listen ofill": "lisinopril",
    # atorvastatin
    "a torvastatin": "atorvastatin",
    "torvastatin": "atorvastatin",
    "atovestatin": "atorvastatin",
    "atavestatin": "atorvastatin",
    "adavastatin": "atorvastatin",
    "atovastatin": "atorvastatin",
    "aatorvastatin": "atorvastatin",
    # metformin
    "met for men": "metformin",
    "met foreman": "metformin",
    "methamorphin": "metformin",
    "metamorphin": "metformin",
    "methomorphin": "metformin",
    "metmorphin": "metformin",
    "met morphin": "metformin",
    # amoxicillin
    "amoxicelin": "amoxicillin",
    "amoxicilin": "amoxicillin",
    "amoxacillin": "amoxicillin",
    # levothyroxine
    "levothoroxin": "levothyroxine",
    "levothoroxine": "levothyroxine",
    "levo thyroxine": "levothyroxine",
    "levothyroxin": "levothyroxine",
    # gabapentin
    "debapentin": "gabapentin",
    "gabapenten": "gabapentin",
    # metoprolol
    "metaprolol": "metoprolol",
    "metoprolo": "metoprolol",
    # warfarin
    "warfare in": "warfarin",
    "warfarein": "warfarin",
    # omeprazole
    "omeprosol": "omeprazole",
    # tirzepatide (Mounjaro/Zepbound)
    "terezipotide": "tirzepatide",
    "oreazipotide": "tirzepatide",
    "terezipatide": "tirzepatide",
    "tirza patide": "tirzepatide",
    "tirzepatite": "tirzepatide",
    "terzapatide": "tirzepatide",
    "erasipotide": "tirzepatide",
    "erezipatide": "tirzepatide",
    "erazipotide": "tirzepatide",
    "rezipatide": "tirzepatide",
    "resipatide": "tirzepatide",
    # semaglutide
    "semaglooted": "semaglutide",
    "semaglatide": "semaglutide",
    "semaglotide": "semaglutide",
    # allergy phrase corrections
    "pine treology": "pine tree allergy",
    "pine treeology": "pine tree allergy",
    "pine tree allergy": "pine tree allergy",
    "peanut allergy": "peanut allergy",
    # spironolactone
    "spiranolactone": "spironolactone",
    # losartan
    "lozartan": "losartan",
    # atenolol
    "a tenolol": "atenolol",
    # escitalopram
    "esatallopram": "escitalopram",
    "esatalopram": "escitalopram",
    # quetiapine
    "quatipine": "quetiapine",
    # risperidone
    "rasparidone": "risperidone",
    "risperdone": "risperidone",
    # montelukast
    "montalucast": "montelukast",
    # doxycycline
    "doxocycline": "doxycycline",
    # 500-drug systematic corrections
    "a backovir": "abacavir",
    "acetretin": "acitretin",
    "a cladinium": "aclidinium",
    "a cyclovir": "acyclovir",
    "atolimiumab": "adalimumab",
    "a dapeline": "adapalene",
    "a fatinib": "afatinib",
    "electinib": "alectinib",
    "alindronate": "alendronate",
    "alfusacin": "alfuzosin",
    "almatriptan": "almotriptan",
    "alaglipton": "alogliptin",
    "amatriptyline": "amitriptyline",
    "a nastrazol": "anastrozole",
    "arefer motorol": "arformoterol",
    "a senipine": "asenapine",
    "adizanivir": "atazanavir",
    "out amoxetine": "atomoxetine",
    "a vanafil": "avanafil",
    "as a thiprene": "azathioprine",
    "astrianam": "aztreonam",
    "baloxivir": "baloxavir",
    "barisitinib": "baricitinib",
    "bechloromethazone": "beclomethasone",
    "benrolizumab": "benralizumab",
    "benzonetate": "benzonatate",
    "beta methazone": "betamethasone",
    "bathanacol": "bethanechol",
    "bisacodil": "bisacodyl",
    "besoprolol": "bisoprolol",
    "bromonadine": "brimonidine",
    "bromocryptine": "bromocriptine",
    "busebrone": "buspirone",
    "candisartan": "candesartan",
    "capicidabine": "capecitabine",
    "captipril": "captopril",
    "carvetolol": "carvedilol",
    "cephaselin": "cefazolin",
    "seftinir": "cefdinir",
    "suffixim": "cefixime",
    "cephrosal": "cefprozil",
    "seftriaxone": "ceftriaxone",
    "suffurizem": "cefuroxime",
    "celacoxib": "celecoxib",
    "serotonib": "ceritinib",
    "sertalizumab": "certolizumab",
    "satirazine": "cetirizine",
    "chlorfiniramine": "chlorpheniramine",
    "colocalciferol": "cholecalciferol",
    "cholesterolamine": "cholestyramine",
    "cyclesinide": "ciclesonide",
    "simetidine": "cimetidine",
    "sinakalset": "cinacalcet",
    "satalapram": "citalopram",
    "clavitipine": "clevidipine",
    "clobatisol": "clobetasol",
    "clomaphen": "clomiphene",
    "calliston": "colistin",
    "crasatinib": "crizotinib",
    "cyanobalamin": "cyanocobalamin",
    "cyclobenzoprene": "cyclobenzaprine",
    "cyclopentylate": "cyclopentolate",
    "debigatran": "dabigatran",
    "daubra fenib": "dabrafenib",
    "dantraline": "dantrolene",
    "dariphenison": "darifenacin",
    "dizipramine": "desipramine",
    "desloratidine": "desloratadine",
    "desigestral": "desogestrel",
    "desvenilifaxine": "desvenlafaxine",
    "dextroemphetamine": "dextroamphetamine",
    "dextremothorfin": "dextromethorphan",
    "dihydroargotamine": "dihydroergotamine",
    "dhaltyazem": "diltiazem",
    "dimonhydrinate": "dimenhydrinate",
    "dolacetron": "dolasetron",
    "dolategravir": "dolutegravir",
    "donepousil": "donepezil",
    "dorsolamide": "dorzolamide",
    "doxopin": "doxepin",
    "drospirinone": "drospirenone",
    "duleglutide": "dulaglutide",
    "dupillumab": "dupilumab",
    "due testoride": "dutasteride",
    "f of irons": "efavirenz",
    "elitriptan": "eletriptan",
    "alvidegravir": "elvitegravir",
    "impagliflozin": "empagliflozin",
    "emtricidabine": "emtricitabine",
    "a nallopril": "enalapril",
    "antacipone": "entacapone",
    "ergatamine": "ergotamine",
    "urtipenem": "ertapenem",
    "s soapicloin": "eszopiclone",
    "a tanner sept": "etanercept",
    "etinogestral": "etonogestrel",
    "exomestine": "exemestane",
    "exenotide": "exenatide",
    "azetomybe": "ezetimibe",
    "famcyclivir": "famciclovir",
    "phaboxostat": "febuxostat",
    "felbomate": "felbamate",
    "phyllozepine": "felodipine",
    "phenafibrate": "fenofibrate",
    "fesoteridine": "fesoterodine",
    "fexofenidine": "fexofenadine",
    "fidexamison": "fidaxomicin",
    "fluid recortisone": "fludrocortisone",
    "flufenizine": "fluphenazine",
    "fluticisone": "fluticasone",
    "formatural": "formoterol",
    "phosphomycin": "fosfomycin",
    "phosphonatoin": "fosphenytoin",
    "provatriptan": "frovatriptan",
    "gephidenab": "gefitinib",
    "gleka-pravir": "glecaprevir",
    "glimeparite": "glimepiride",
    "gliposide": "glipizide",
    "gliburide": "glyburide",
    "glycoprolate": "glycopyrrolate",
    "golemiamab": "golimumab",
    "granicetron": "granisetron",
    "guifenicin": "guaifenesin",
    "haloparadol": "haloperidol",
    "hydrolozin": "hydralazine",
    "hydroxazine": "hydroxyzine",
    "a brutinib": "ibrutinib",
    "idelilisib": "idelalisib",
    "isoparadone": "iloperidone",
    "a matinib": "imatinib",
    "amipramine": "imipramine",
    "endomethacin": "indomethacin",
    "iprotropium": "ipratropium",
    "herbacartin": "irbesartan",
    "is ratapine": "isradipine",
    "atroconazole": "itraconazole",
    "keturalak": "ketorolac",
    "lakosemide": "lacosamide",
    "lactilose": "lactulose",
    "landsoprazole": "lansoprazole",
    "latinoprost": "latanoprost",
    "leflunamide": "leflunomide",
    "linalidomide": "lenalidomide",
    "letrasol": "letrozole",
    "levabuteral": "levalbuterol",
    "levataracetam": "levetiracetam",
    "levomilnasipran": "levomilnacipran",
    "levinor gestural": "levonorgestrel",
    "linoclotide": "linaclotide",
    "linoglipatin": "linagliptin",
    "linkamycin": "lincomycin",
    "linesolid": "linezolid",
    "lyothyronine": "liothyronine",
    "laraglotide": "liraglutide",
    "listecsamphetamine": "lisdexamfetamine",
    "loparamide": "loperamide",
    "lopinivir": "lopinavir",
    "loratidine": "loratadine",
    "larazepam": "lorazepam",
    "lorcasarin": "lorcaserin",
    "laracidone": "lurasidone",
    "meclosine": "meclizine",
    "madroxyprogesterone": "medroxyprogesterone",
    "maloxicam": "meloxicam",
    "mepillazimab": "mepolizumab",
    "memprobamate": "meprobamate",
    "methimizol": "methimazole",
    "methacarbamol": "methocarbamol",
    "mykonazole": "miconazole",
    "medazolam": "midazolam",
    "miglital": "miglitol",
    "milanacipran": "milnacipran",
    "mirabegrawn": "mirabegron",
    "mirtazepine": "mirtazapine",
    "mysoprostol": "misoprostol",
    "momentosone": "mometasone",
    "mycofinalate": "mycophenolate",
    "nidolol": "nadolol",
    "near atriptan": "naratriptan",
    "nataglionide": "nateglinide",
    "nephazadone": "nefazodone",
    "necrotipine": "nicardipine",
    "nephetapine": "nifedipine",
    "nicetadine": "nizatidine",
    "norethandrone": "norethindrone",
    "norefloxacin": "norfloxacin",
    "nistatin": "nystatin",
    "octriotide": "octreotide",
    "ulmusartin": "olmesartan",
    "omelizimab": "omalizumab",
    "on dancitron": "ondansetron",
    "oralestate": "orlistat",
    "orphanedrin": "orphenadrine",
    "osaltamovir": "oseltamivir",
    "osemerdinib": "osimertinib",
    "oxgerbazepine": "oxcarbazepine",
    "oxybutanin": "oxybutynin",
    "palaparidone": "paliperidone",
    "peroxetine": "paroxetine",
    "parampanol": "perampanel",
    "perfenicine": "perphenazine",
    "phenolzine": "phenelzine",
    "phenatoin": "phenytoin",
    "pilacarpine": "pilocarpine",
    "pymicrolimus": "pimecrolimus",
    "pyaglidazone": "pioglitazone",
    "paroxicam": "piroxicam",
    "plecanotide": "plecanatide",
    "posiconazole": "posaconazole",
    "pramipexil": "pramipexole",
    "pramlentide": "pramlintide",
    "primadone": "primidone",
    "probenesade": "probenecid",
    "propyethiurusil": "propylthiouracil",
    "cillium": "psyllium",
    "quinopril": "quinapril",
    "rebepprizole": "rabeprazole",
    "reloxapine": "raloxifene",
    "remeltiaun": "ramelteon",
    "renytidine": "ranitidine",
    "resagilline": "rasagiline",
    "rapaglionide": "repaglinide",
    "refaxamen": "rifaximin",
    "realpavirin": "rilpivirine",
    "resedrinate": "risedronate",
    "rhizotriptan": "rizatriptan",
    "roflumelast": "roflumilast",
    "rapineral": "ropinirole",
    "rosoglitazone": "rosiglitazone",
    "resuvastatin": "rosuvastatin",
    "rotigatin": "rotigotine",
    "ruffinimide": "rufinamide",
    "selediline": "selegiline",
    "sena": "senna",
    "novellumir": "sevelamer",
    "soldenifol": "sildenafil",
    "silidosin": "silodosin",
    "citaglipton": "sitagliptin",
    "sofaspivir": "sofosbuvir",
    "solifenosin": "solifenacin",
    "sucralphate": "sucralfate",
    "salindac": "sulindac",
    "suvareksant": "suvorexant",
    "takrolimus": "tacrolimus",
    "to dhalafil": "tadalafil",
    "tamcilocin": "tamsulosin",
    "to pentadol": "tapentadol",
    "tizaratine": "tazarotene",
    "tetazolid": "tedizolid",
    "telmasartin": "telmisartan",
    "tamazepam": "temazepam",
    "temizolamide": "temozolomide",
    "tonofovir": "tenofovir",
    "terrazinc": "terazosin",
    "turbinopin": "terbinafine",
    "theophelin": "theophylline",
    "thioretizine": "thioridazine",
    "tyagabine": "tiagabine",
    "tycarcillin": "ticarcillin",
    "timeallol": "timolol",
    "tinnitusol": "tinidazole",
    "teotropium": "tiotropium",
    "tosillizumab": "tocilizumab",
    "tophacidinib": "tofacitinib",
    "tulipone": "tolcapone",
    "tulteridine": "tolterodine",
    "to pyramid": "topiramate",
    "tranlylcypramine": "tranylcypromine",
    "trasodone": "trazodone",
    "tritinoin": "tretinoin",
    "trimetoprim": "trimethoprim",
    "trimiprimine": "trimipramine",
    "upatisidinib": "upadacitinib",
    "use to kinumab": "ustekinumab",
    "valicyclovir": "valacyclovir",
    "velproate": "valproate",
    "balsartin": "valsartan",
    "venkamaisan": "vancomycin",
    "fardenophil": "vardenafil",
    "vettelizumab": "vedolizumab",
    "vemurafinib": "vemurafenib",
    "varapamil": "verapamil",
    "valazadone": "vilazodone",
    "voracanazole": "voriconazole",
    "cephalochast": "zafirlukast",
    "salipline": "zaleplon",
    "cinamovir": "zanamivir",
    "sudoviodine": "zidovudine",
    "silutin": "zileuton",
    "supracedone": "ziprasidone",
    "soladronic": "zoledronic",
    "solmatriptan": "zolmitriptan",
    "solpidem": "zolpidem",
    "senisemide": "zonisamide",
}

DRUGS = [
    "lisinopril", "atorvastatin", "amlodipine", "metoprolol", "losartan",
    "hydrochlorothiazide", "furosemide", "carvedilol", "spironolactone",
    "digoxin", "amiodarone", "diltiazem", "verapamil", "nifedipine",
    "hydralazine", "clonidine", "doxazosin", "terazosin", "prazosin",
    "isosorbide", "nitroglycerin", "apixaban", "rivaroxaban", "dabigatran",
    "warfarin", "clopidogrel", "ticagrelor", "enoxaparin", "aspirin",
    "rosuvastatin", "simvastatin", "pravastatin", "fluvastatin", "lovastatin",
    "ezetimibe", "fenofibrate", "gemfibrozil", "cholestyramine",
    "atenolol", "bisoprolol", "nadolol", "propranolol", "timolol",
    "ramipril", "enalapril", "captopril", "benazepril", "quinapril",
    "valsartan", "olmesartan", "irbesartan", "candesartan", "telmisartan",
    "felodipine", "nicardipine", "isradipine",
    "metformin", "insulin", "glipizide", "glimepiride", "glyburide",
    "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "empagliflozin", "canagliflozin", "dapagliflozin", "ertugliflozin",
    "liraglutide", "semaglutide", "dulaglutide", "exenatide", "tirzepatide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "repaglinide", "nateglinide", "pramlintide",
    "levothyroxine", "liothyronine", "methimazole", "propylthiouracil",
    "albuterol", "levalbuterol", "salmeterol", "formoterol",
    "tiotropium", "ipratropium", "aclidinium", "umeclidinium",
    "fluticasone", "budesonide", "beclomethasone", "mometasone",
    "montelukast", "zafirlukast", "zileuton", "theophylline",
    "amoxicillin", "ampicillin", "dicloxacillin", "nafcillin",
    "penicillin", "piperacillin", "azithromycin", "clarithromycin",
    "erythromycin", "fidaxomicin", "ciprofloxacin", "levofloxacin",
    "moxifloxacin", "norfloxacin", "doxycycline", "minocycline",
    "tetracycline", "trimethoprim", "sulfamethoxazole", "nitrofurantoin",
    "metronidazole", "tinidazole", "clindamycin", "vancomycin",
    "daptomycin", "linezolid", "tedizolid", "ceftriaxone", "cefazolin",
    "cephalexin", "cefdinir", "cefuroxime", "cefprozil", "cefixime",
    "meropenem", "imipenem", "ertapenem", "aztreonam",
    "ibuprofen", "naproxen", "celecoxib", "diclofenac", "indomethacin",
    "ketorolac", "meloxicam", "piroxicam", "etodolac", "sulindac",
    "acetaminophen", "tramadol", "codeine", "hydrocodone", "oxycodone",
    "oxymorphone", "hydromorphone", "morphine", "fentanyl",
    "buprenorphine", "methadone", "naloxone", "naltrexone", "tapentadol",
    "pregabalin", "gabapentin", "duloxetine", "milnacipran",
    "cyclobenzaprine", "baclofen", "tizanidine", "carisoprodol",
    "methocarbamol", "orphenadrine",
    "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
    "triamcinolone", "betamethasone", "hydrocortisone", "fludrocortisone",
    "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
    "fluvoxamine", "venlafaxine", "desvenlafaxine", "levomilnacipran",
    "bupropion", "mirtazapine", "trazodone", "nefazodone",
    "vilazodone", "vortioxetine", "amitriptyline", "nortriptyline",
    "imipramine", "desipramine", "clomipramine", "doxepin",
    "phenelzine", "tranylcypromine", "selegiline",
    "quetiapine", "olanzapine", "risperidone", "aripiprazole",
    "ziprasidone", "lurasidone", "asenapine", "iloperidone",
    "paliperidone", "clozapine", "haloperidol", "fluphenazine",
    "perphenazine", "thioridazine",
    "lithium", "valproate", "lamotrigine", "carbamazepine", "oxcarbazepine",
    "alprazolam", "lorazepam", "clonazepam", "diazepam", "temazepam",
    "triazolam", "midazolam", "zolpidem", "zaleplon", "eszopiclone",
    "ramelteon", "suvorexant", "hydroxyzine", "buspirone",
    "amphetamine", "methylphenidate", "lisdexamfetamine",
    "dextroamphetamine", "atomoxetine", "guanfacine",
    "levetiracetam", "topiramate", "zonisamide", "lacosamide",
    "perampanel", "phenytoin", "fosphenytoin", "phenobarbital",
    "primidone", "ethosuximide", "vigabatrin", "tiagabine",
    "felbamate", "rufinamide", "clobazam",
    "sumatriptan", "rizatriptan", "zolmitriptan", "naratriptan",
    "eletriptan", "frovatriptan", "almotriptan", "ergotamine",
    "donepezil", "rivastigmine", "galantamine", "memantine",
    "carbidopa", "levodopa", "pramipexole", "ropinirole",
    "rotigotine", "rasagiline", "entacapone", "tolcapone", "amantadine",
    "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole",
    "rabeprazole", "ranitidine", "famotidine", "cimetidine",
    "nizatidine", "sucralfate", "misoprostol", "metoclopramide",
    "ondansetron", "granisetron", "dolasetron", "prochlorperazine",
    "promethazine", "meclizine", "dimenhydrinate", "loperamide",
    "mesalamine", "sulfasalazine", "balsalazide", "olsalazine",
    "lactulose", "bisacodyl", "senna", "docusate", "psyllium",
    "linaclotide", "lubiprostone", "plecanatide", "rifaximin",
    "tamsulosin", "alfuzosin", "silodosin",
    "finasteride", "dutasteride", "tadalafil", "sildenafil",
    "vardenafil", "avanafil", "oxybutynin", "tolterodine",
    "solifenacin", "darifenacin", "fesoterodine", "mirabegron",
    "estradiol", "progesterone", "medroxyprogesterone", "norethindrone",
    "levonorgestrel", "desogestrel", "etonogestrel", "drospirenone",
    "raloxifene", "tamoxifen", "anastrozole", "letrozole",
    "exemestane", "clomiphene",
    "tretinoin", "adapalene", "tazarotene", "isotretinoin", "acitretin",
    "clobetasol", "tacrolimus", "pimecrolimus", "ivermectin", "permethrin",
    "latanoprost", "bimatoprost", "travoprost", "dorzolamide",
    "brimonidine", "pilocarpine",
    "acyclovir", "valacyclovir", "famciclovir", "oseltamivir",
    "zanamivir", "baloxavir", "sofosbuvir", "ledipasvir",
    "velpatasvir", "glecaprevir", "pibrentasvir", "tenofovir",
    "emtricitabine", "abacavir", "lamivudine", "zidovudine",
    "efavirenz", "rilpivirine", "dolutegravir", "raltegravir",
    "elvitegravir", "atazanavir", "darunavir", "lopinavir",
    "fluconazole", "itraconazole", "voriconazole", "posaconazole",
    "terbinafine", "clotrimazole", "miconazole", "nystatin",
    "methotrexate", "hydroxychloroquine", "leflunomide",
    "azathioprine", "mycophenolate", "cyclosporine",
    "adalimumab", "etanercept", "infliximab", "abatacept",
    "rituximab", "tocilizumab", "baricitinib", "tofacitinib",
    "upadacitinib",
    "imatinib", "erlotinib", "gefitinib", "afatinib", "osimertinib",
    "crizotinib", "ceritinib", "alectinib", "dabrafenib", "vemurafenib",
    "ibrutinib", "idelalisib", "venetoclax", "lenalidomide",
    "capecitabine", "temozolomide", "hydroxyurea",
    "allopurinol", "febuxostat", "colchicine", "probenecid",
    "alendronate", "risedronate", "ibandronate", "zoledronic",
    "calcitriol", "cholecalciferol",
    "desmopressin", "octreotide", "bromocriptine", "cabergoline",
    "cinacalcet", "sevelamer", "phentermine", "orlistat",
    "melatonin", "diphenhydramine", "loratadine", "cetirizine",
    "fexofenadine", "levocetirizine", "desloratadine",
    "chlorpheniramine", "pseudoephedrine", "phenylephrine",
    "oxymetazoline", "guaifenesin", "dextromethorphan", "benzonatate",
]

SYMPTOMS = [
    "chest pain", "shortness of breath", "dyspnea", "dizziness",
    "fatigue", "weakness", "nausea", "vomiting", "headache",
    "fever", "chills", "cough", "edema", "swelling", "back pain",
    "joint pain", "anxiety", "insomnia", "confusion", "weight loss",
    "abdominal pain", "diarrhea", "constipation", "palpitations",
    "syncope", "numbness", "tingling", "rash", "itching",
    "polyuria", "polydipsia", "hematuria", "dysuria",
    "side effects", "pain", "discomfort", "inflammation",
    "bloating", "heartburn", "reflux", "indigestion",
    "muscle pain", "muscle weakness", "cramps", "tremor",
    "blurred vision", "dry mouth", "hair loss", "weight gain",
    "depression", "mood changes", "memory loss", "brain fog",
]


def extract_entities(text):
    t = text.lower()

    # Apply phonetic corrections
    for wrong, right in PHONETIC_CORRECTIONS.items():
        t = t.replace(wrong, right)

    # Extract specific allergy patterns only
    allergy_noun = re.findall(
        r'\b(?:pine\s+tree|peanut|tree\s+nut|nut|penicillin|sulfa|latex|'
        r'shellfish|gluten|dairy|egg|wheat|soy|fish|bee|wasp|mold|dust|'
        r'pollen|cat|dog|grass|ragweed|seasonal|drug|food|environmental|'
        r'aspirin|ibuprofen|codeine)\s+allerg(?:y|ies)',
        t
    )

    # Extract vitals
    weight_lbs   = re.findall(r'(\d{2,3})\s*pounds?', t)
    height       = re.findall(r'(\d)\s*(?:feet?|foot)\s*(?:and\s*)?(\d{1,2})\s*inches?', t)
    weight_lost  = re.findall(r'lost\s*(\d+)\s*pounds?', t)
    goal_weight  = re.findall(r'goal\s*weight[^.]*?(\d+)\s*pounds?', t)

    vitals = {}
    if weight_lbs:
        vitals["weight_lbs"] = weight_lbs[0] + " lbs"
    if height:
        h = height[0]
        vitals["height"] = f"{h[0]}'{h[1]}\""
    if weight_lost:
        vitals["weight_lost"] = weight_lost[0] + " lbs"
    if goal_weight:
        vitals["goal_weight"] = goal_weight[0] + " lbs"

    return {
        "drugs":     [d for d in DRUGS if d in t],
        "symptoms":  [s for s in SYMPTOMS if s in t],
        "dosages":   re.findall(
            r'\d+\.?\d*\s*(?:mg|mcg|ml|milligrams?|micrograms?|units?)', t),
        "frequency": re.findall(
            r'once\s+(?:a\s+)?daily'
            r'|twice\s+(?:a\s+)?daily'
            r'|three\s+times\s+(?:a\s+)?(?:day|daily)'
            r'|every\s+\d+\s+hours?'
            r'|at\s+bedtime'
            r'|as\s+needed'
            r'|with\s+meals?'
            r'|every\s+morning'
            r'|every\s+night', t),
        "allergies": list(dict.fromkeys([a.strip() for a in allergy_noun])),
        "vitals":    vitals,
        "corrected": t,
    }


def degrade_to_telephony(input_path, output_path):
    import subprocess
    tmp = output_path + "_tmp.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "8000", "-ac", "1",
        "-acodec", "pcm_s16le", tmp
    ], capture_output=True, check=True)
    audio, sr = sf.read(tmp)
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    ulaw = audioop.lin2ulaw(audio_int16, 2)
    decoded = audioop.ulaw2lin(ulaw, 2)
    audio_out = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32767
    sf.write(output_path, audio_out, 8000)
    os.remove(tmp)
    return output_path


def run_pipeline(audio_path: str) -> dict:
    # Base Whisper transcription
    base_text = base_model.transcribe(audio_path)["text"]

    # Fine-tuned MedEar — only for short clips (under 30s)
    # Long conversational audio uses base model for better accuracy
    audio = whisper.load_audio(audio_path)
    duration = len(audio) / 16000

    if duration > 30:
        # Long audio: use base model transcription, apply corrections via NER
        ft_text = base_text
    else:
        # Short clip: use fine-tuned model
        audio_trimmed = whisper.pad_or_trim(audio)
        features = processor(
            audio_trimmed, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        prompt_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        with torch.no_grad():
            ids = model.generate(features, forced_decoder_ids=prompt_ids)
        ft_text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    # Run NER on both transcripts and merge results
    entities_ft   = extract_entities(ft_text)
    entities_base = extract_entities(base_text)
    corrected     = entities_ft.pop("corrected")
    entities_base.pop("corrected")

    # Merge vitals — prefer base for long audio
    merged_vitals = {**entities_ft.get("vitals", {}), **entities_base.get("vitals", {})}

    merged = {
        "drugs":    list(dict.fromkeys(
            entities_ft["drugs"]    + entities_base["drugs"])),
        "symptoms": list(dict.fromkeys(
            entities_ft["symptoms"] + entities_base["symptoms"])),
        "dosages":  list(dict.fromkeys(
            entities_ft["dosages"]  + entities_base["dosages"])),
        "frequency": list(dict.fromkeys(
            entities_ft["frequency"] + entities_base["frequency"])),
        "allergies": list(dict.fromkeys(
            entities_ft["allergies"] + entities_base["allergies"])),
        "vitals": merged_vitals,
    }

    return {
        "base_transcript":      base_text,
        "medear_transcript":    ft_text,
        "corrected_transcript": corrected,
        "entities":             merged,
    }