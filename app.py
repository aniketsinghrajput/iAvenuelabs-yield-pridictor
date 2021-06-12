#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn.preprocessing import StandardScaler




# In[2]:


app= Flask(__name__)
model=pickle.load(open('model.pkl','rb'))



# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


State_dict={'Chandigarh': 0,
 'Mizoram': 1,
 'Arunanchal Pradesh': 2,
 'Sikkim': 3,
 'Nagaland': 4,
 'Meghalaya': 5,
 'Manipur': 6,
 'Dadara & Nagar Havelli': 7,
 'Himachal Pradesh': 8,
 'Tripura': 9,
 'Jharkhand': 10,
 'Jammu & Kashmir': 11,
 'Chhattisgarh': 12,
 'Odisha': 13,
 'Madhya Pradesh': 14,
 'Bihar': 15,
 'Rajasthan': 16,
 'Uttarakhand': 17,
 'Karnataka': 18,
 'Gujarat': 19,
 'Telangana': 20,
 'Haryana': 21,
 'Maharashtra': 22,
 'Uttar Pradesh': 23,
 'Assam': 24,
 'West Bengal': 25,
 'Punjab': 26,
 'Puducherry': 27,
 'Tamil Nadu': 28,
 'Andhra Pradesh': 29,
 'Goa': 30,
 'Andaman & Nicobar Island': 31,
 'Kerala': 32,
 'Andamanandnicobarislands': 31,
 'Andhrapradesh': 29,
 'Arunachalpradesh': 2,
 'Dadraandnagarhavelli': 7,
 'Himachalpradesh': 8,
 'Jammuandkashmir': 11,
 'Madhyapradesh': 14,
 'Tamilnadu': 28,
 'Uttarpradesh': 23,
 'Westbengal': 25}


# In[5]:


Dist_dict={'MUMBAI': 0,
 'MAMIT': 1,
 'LEH LADAKH': 2,
 'KINNAUR': 3,
 'LAWNGTLAI': 4,
 'KARGIL': 5,
 'HYDERABAD': 6,
 'CHANDIGARH': 7,
 'KURUNG KUMEY': 8,
 'NAMSAI': 9,
 'SAIHA': 10,
 'KHUNTI': 11,
 'NORTH DISTRICT': 12,
 'RAMGARH': 13,
 'SOUTH GARO HILLS': 14,
 'ANJAW': 15,
 'NORTH GARO HILLS': 16,
 'WEST JAINTIA HILLS': 17,
 'TAWANG': 18,
 'NARAYANPUR': 19,
 'SHOPIAN': 20,
 'LUNGLEI': 21,
 'WEST KAMENG': 22,
 'TIRAP': 23,
 'SOUTH WEST KHASI HILLS': 24,
 'CHAMPHAI': 25,
 'LONGLENG': 26,
 'PALGHAR': 27,
 'UPPER SIANG': 28,
 'UPPER SUBANSIRI': 29,
 'KISHTWAR': 30,
 'EAST KAMENG': 31,
 'LONGDING': 32,
 'KOLASIB': 33,
 'TAMENGLONG': 34,
 'SERCHHIP': 35,
 'CHAMPAWAT': 36,
 'CHANDEL': 37,
 'SRINAGAR': 38,
 'DIBANG VALLEY': 39,
 'KIPHIRE': 40,
 'LOWER SUBANSIRI': 41,
 'EAST JAINTIA HILLS': 42,
 'RUDRA PRAYAG': 43,
 'MON': 44,
 'PAPUM PARE': 45,
 'LAHUL AND SPITI': 46,
 'ZUNHEBOTO': 47,
 'KONDAGAON': 48,
 'MOKOKCHUNG': 49,
 'SOUTH WEST GARO HILLS': 50,
 'BANDIPORA': 51,
 'KODERMA': 52,
 'ARWAL': 53,
 'CHANGLANG': 54,
 'LATEHAR': 55,
 'BAGESHWAR': 56,
 'EAST GARO HILLS': 57,
 'KOHIMA': 58,
 'PEREN': 59,
 'UMARIA': 60,
 'REASI': 61,
 'UKHRUL': 62,
 'PHEK': 63,
 'AIZAWL': 64,
 'RAMBAN': 65,
 'WOKHA': 66,
 'WEST DISTRICT': 67,
 'BOKARO': 68,
 'TUENSANG': 69,
 'EAST DISTRICT': 70,
 'WEST SIANG': 71,
 'DEOGARH': 72,
 'RI BHOI': 73,
 'KOREA': 74,
 'WEST KHASI HILLS': 75,
 'DHALAI': 76,
 'KANDHAMAL': 77,
 'UNAKOTI': 78,
 'KULGAM': 79,
 'GANDERBAL': 80,
 'BENGALURU URBAN': 81,
 'GARHWA': 82,
 'GAJAPATI': 83,
 'RAYAGADA': 84,
 'CHURACHANDPUR': 85,
 'MUNGER': 86,
 'SAHEBGANJ': 87,
 'LOWER DIBANG VALLEY': 88,
 'JHARSUGUDA': 89,
 'ANUPPUR': 90,
 'SHIMLA': 91,
 'SENAPATI': 92,
 'UTTAR KASHI': 93,
 'KORBA': 94,
 'BOUDH': 95,
 'SOUTH DISTRICT': 96,
 'DANG': 97,
 'DHANBAD': 98,
 'EAST SIANG': 99,
 'PITHORAGARH': 100,
 'BISHNUPUR': 101,
 'NUAPADA': 102,
 'NORTH TRIPURA': 103,
 'LOHARDAGA': 104,
 'LOHIT': 105,
 'KHOWAI': 106,
 'THE NILGIRIS': 107,
 'CHATRA': 108,
 'PULWAMA': 109,
 'CHAMBA': 110,
 'PAURI GARHWAL': 111,
 'AGAR MALWA': 112,
 'KULLU': 113,
 'SHEIKHPURA': 114,
 'THOUBAL': 115,
 'SOLAN': 116,
 'BADGAM': 117,
 'JASHPUR': 118,
 'RAJSAMAND': 119,
 'ALIRAJPUR': 120,
 'IMPHAL WEST': 121,
 'SURAJPUR': 122,
 'SIMDEGA': 123,
 'TEHRI GARHWAL': 124,
 'JAMUI': 125,
 'SARAIKELA KHARSAWAN': 126,
 'JAISALMER': 127,
 'JAMTARA': 128,
 'IMPHAL EAST': 129,
 'PAKUR': 130,
 'DANTEWADA': 131,
 'SIROHI': 132,
 'GODDA': 133,
 'DINDORI': 134,
 'SAMBA': 135,
 'Dadara & Nagar Havelli': 136,
 'DUNGARPUR': 137,
 'BALODA BAZAR': 138,
 'PALAMU': 139,
 'NAWADA': 140,
 'SHEOHAR': 141,
 'ANUGUL': 142,
 'DODA': 143,
 'CHAMOLI': 144,
 'KABIRDHAM': 145,
 'UNA': 146,
 'CHIKBALLAPUR': 147,
 'ALMORA': 148,
 'RAIGARH': 149,
 'SUKMA': 150,
 'SHAHDOL': 151,
 'SIRMAUR': 152,
 'GOMATI': 153,
 'DIMAPUR': 154,
 'WEST GARO HILLS': 155,
 'KHAGARIA': 156,
 'MALKANGIRI': 157,
 'PANCHKULA': 158,
 'MANDLA': 159,
 'GARIYABAND': 160,
 'KODAGU': 161,
 'KATHUA': 162,
 'JEHANABAD': 163,
 'KUPWARA': 164,
 'BASTAR': 165,
 'EAST KHASI HILLS': 166,
 'SINGRAULI': 167,
 'KATNI': 168,
 'LAKHISARAI': 169,
 'KHORDHA': 170,
 'PANNA': 171,
 'KANKER': 172,
 'DHAMTARI': 173,
 'RAMANAGARA': 174,
 'GUMLA': 175,
 'TINSUKIA': 176,
 'CHITRAKOOT': 177,
 'AJMER': 178,
 'SIDHI': 179,
 'DHENKANAL': 180,
 'JAJAPUR': 181,
 'SOUTH TRIPURA': 182,
 'BARWANI': 183,
 'WEST TRIPURA': 184,
 'UDHAMPUR': 185,
 'SAMBALPUR': 186,
 'HAZARIBAGH': 187,
 'DEOGHAR': 188,
 'DIMA HASAO': 189,
 'JHABUA': 190,
 'GIRIDIH': 191,
 'SONEPUR': 192,
 'BEGUSARAI': 193,
 'TIKAMGARH': 194,
 'JAGATSINGHAPUR': 195,
 'KENDUJHAR': 196,
 'GADCHIROLI': 197,
 'MUNGELI': 198,
 'MADHUBANI': 199,
 'BANGALORE RURAL': 200,
 'PALI': 201,
 'SURGUJA': 202,
 'UTTAR KANNAD': 203,
 'MAHOBA': 204,
 'SONBHADRA': 205,
 'SEPAHIJALA': 206,
 'DUMKA': 207,
 'SANT RAVIDAS NAGAR': 208,
 'BARAMULLA': 209,
 'VAISHALI': 210,
 'YADGIR': 211,
 'BANSWARA': 212,
 'BHOPAL': 213,
 'SAHARSA': 214,
 'BALOD': 215,
 'MAHASAMUND': 216,
 'DARBHANGA': 217,
 'WEST SINGHBHUM': 218,
 'NEEMUCH': 219,
 'NAYAGARH': 220,
 'RAJNANDGAON': 221,
 'BILASPUR': 222,
 'KOLAR': 223,
 'SUNDARGARH': 224,
 'GAYA': 225,
 'SATNA': 226,
 'JANJGIR-CHAMPA': 227,
 'GONDIA': 228,
 'KENDRAPARA': 229,
 'UDAIPUR': 230,
 'KANGRA': 231,
 'POONCH': 232,
 'DAMOH': 233,
 'ANANTNAG': 234,
 'RAJAURI': 235,
 'KISHANGANJ': 236,
 'BHAGALPUR': 237,
 'BALANGIR': 238,
 'GADAG': 239,
 'JAMMU': 240,
 'JALORE': 241,
 'RANGAREDDI': 242,
 'MANDI': 243,
 'DIBRUGARH': 244,
 'BALAGHAT': 245,
 'BEMETARA': 246,
 'HAMIRPUR': 247,
 'NALANDA': 248,
 'SARAN': 249,
 'CHIKMAGALUR': 250,
 'BARMER': 251,
 'KORAPUT': 252,
 'UDUPI': 253,
 'NABARANGPUR': 254,
 'REWA': 255,
 'PURI': 256,
 'PANCH MAHALS': 257,
 'DHARWAD': 258,
 'DHEMAJI': 259,
 'SHEOPUR': 260,
 'BANKA': 261,
 'KAUSHAMBI': 262,
 'JABALPUR': 263,
 'EAST SINGHBUM': 264,
 'SEONI': 265,
 'SUPAUL': 266,
 'BUXAR': 267,
 'KOPPAL': 268,
 'CHHATARPUR': 269,
 'KALAHANDI': 270,
 'BHILWARA': 271,
 'NAINITAL': 272,
 'LUCKNOW': 273,
 'SAWAI MADHOPUR': 274,
 'MADHEPURA': 275,
 'BHIND': 276,
 'MUZAFFARPUR': 277,
 'KHARGONE': 278,
 'DHOLPUR': 279,
 'BANDA': 280,
 'PATNA': 281,
 'CUTTACK': 282,
 'DATIA': 283,
 'BALESHWAR': 284,
 'BHOJPUR': 285,
 'GANDHINAGAR': 286,
 'TONK': 287,
 'PORBANDAR': 288,
 'BHADRAK': 289,
 'SAMASTIPUR': 290,
 'ASHOKNAGAR': 291,
 'SAGAR': 292,
 'LALITPUR': 293,
 'HAVERI': 294,
 'HARDA': 295,
 'SIWAN': 296,
 'RAIPUR': 297,
 'KARAULI': 298,
 'DAUSA': 299,
 'PURULIA': 300,
 'GWALIOR': 301,
 'SINDHUDURG': 302,
 'KHANDWA': 303,
 'KAIMUR (BHABUA)': 304,
 'JHANSI': 305,
 'MAYURBHANJ': 306,
 'DARJEELING': 307,
 'RAJGARH': 308,
 'GUNA': 309,
 'RANCHI': 310,
 'KATIHAR': 311,
 'BIKANER': 312,
 'DOHAD': 313,
 'JHALAWAR': 314,
 'CHITRADURGA': 315,
 'THANE': 316,
 'ADILABAD': 317,
 'ARARIA': 318,
 'DURG': 319,
 'RATNAGIRI': 320,
 'RATLAM': 321,
 'MIRZAPUR': 322,
 'BELLARY': 323,
 'PATAN': 324,
 'RAICHUR': 325,
 'SHRAVASTI': 326,
 'WASHIM': 327,
 'BETUL': 328,
 'VARANASI': 329,
 'HAILAKANDI': 330,
 'PURNIA': 331,
 'SHIVPURI': 332,
 'MANDSAUR': 333,
 'CHANDAULI': 334,
 'GANJAM': 335,
 'BARGARH': 336,
 'RAISEN': 337,
 'BUNDI': 338,
 'BURHANPUR': 339,
 'KOTA': 340,
 'BHANDARA': 341,
 'MORENA': 342,
 'BARAN': 343,
 'JODHPUR': 344,
 'CHANDRAPUR': 345,
 'INDORE': 346,
 'DEHRADUN': 347,
 'SITAMARHI': 348,
 'CHAMARAJANAGAR': 349,
 'CHURU': 350,
 'CHITTORGARH': 351,
 'DAKSHIN KANNAD': 352,
 'VIDISHA': 353,
 'KAMRUP METRO': 354,
 'PRATAPGARH': 355,
 'GAUTAM BUDDHA NAGAR': 356,
 'SIKAR': 357,
 'TUMKUR': 358,
 'ETAWAH': 359,
 'HOSHANGABAD': 360,
 'AMETHI': 361,
 'SANT KABEER NAGAR': 362,
 'SEHORE': 363,
 'SHAJAPUR': 364,
 'RAIGAD': 365,
 'JALAUN': 366,
 'KADAPA': 367,
 'BHARATPUR': 368,
 'NAGPUR': 369,
 'REWARI': 370,
 'WARDHA': 371,
 'KANPUR DEHAT': 372,
 'AURAIYA': 373,
 'DHAR': 374,
 'AKOLA': 375,
 'JHUNJHUNU': 376,
 'ROHTAS': 377,
 'DEWAS': 378,
 'MAHBUBNAGAR': 379,
 'UNNAO': 380,
 'DHULE': 381,
 'PURBI CHAMPARAN': 382,
 'HASSAN': 383,
 'UJJAIN': 384,
 'MAHESANA': 385,
 'RAE BARELI': 386,
 'SHIMOGA': 387,
 'NANDURBAR': 388,
 'KACHCHH': 389,
 'MAU': 390,
 'AHMADABAD': 391,
 'KANPUR NAGAR': 392,
 'MYSORE': 393,
 'KHEDA': 394,
 'LAKHIMPUR': 395,
 'AMRAVATI': 396,
 'SABAR KANTHA': 397,
 'NAGAUR': 398,
 'GULBARGA': 399,
 'PRAKASAM': 400,
 'KASGANJ': 401,
 'CHHINDWARA': 402,
 'WARANGAL': 403,
 'GURGAON': 404,
 'DAVANGERE': 405,
 'GORAKHPUR': 406,
 'SIDDHARTH NAGAR': 407,
 'GOPALGANJ': 408,
 'NARSINGHPUR': 409,
 'JALPAIGURI': 410,
 'MAHENDRAGARH': 411,
 'VADODARA': 412,
 'BULDHANA': 413,
 'FATEHPUR': 414,
 'NALGONDA': 415,
 'JHAJJAR': 416,
 'HATHRAS': 417,
 'TAPI': 418,
 'JAIPUR': 419,
 'ALLAHABAD': 420,
 'MAINPURI': 421,
 'KURNOOL': 422,
 'HINGOLI': 423,
 'S.A.S NAGAR': 424,
 'BALLIA': 425,
 'FARIDABAD': 426,
 'BIDAR': 427,
 'NARMADA': 428,
 'HANUMANGARH': 429,
 'MEWAT': 430,
 'SULTANPUR': 431,
 'DINAJPUR UTTAR': 432,
 'BIJAPUR': 433,
 'GHAZIPUR': 434,
 'AURANGABAD': 435,
 'NANDED': 436,
 'DINAJPUR DAKSHIN': 437,
 'CHIRANG': 438,
 'AMBEDKAR NAGAR': 439,
 'ALWAR': 440,
 'KANNAUJ': 441,
 'GANGANAGAR': 442,
 'JAMNAGAR': 443,
 'DEORIA': 444,
 'ANAND': 445,
 'FIROZABAD': 446,
 'YAVATMAL': 447,
 'BANAS KANTHA': 448,
 'AMRELI': 449,
 'PARBHANI': 450,
 'MATHURA': 451,
 'KOKRAJHAR': 452,
 'NIZAMABAD': 453,
 'FAIZABAD': 454,
 'PANIPAT': 455,
 'KARBI ANGLONG': 456,
 'CACHAR': 457,
 'JALNA': 458,
 'OSMANABAD': 459,
 'FARRUKHABAD': 460,
 'SURENDRANAGAR': 461,
 'ETAH': 462,
 'KARIMGANJ': 463,
 'KARIMNAGAR': 464,
 'MANDYA': 465,
 'PATHANKOT': 466,
 'JAUNPUR': 467,
 'SIVASAGAR': 468,
 'PALWAL': 469,
 'BANKURA': 470,
 'BAHRAICH': 471,
 'MALDAH': 472,
 'BARABANKI': 473,
 'AZAMGARH': 474,
 'BHIWANI': 475,
 'ROHTAK': 476,
 'NAVSARI': 477,
 'UDALGURI': 478,
 'MAHARAJGANJ': 479,
 'HISAR': 480,
 'VALSAD': 481,
 'ALIGARH': 482,
 'ARIYALUR': 483,
 'RUPNAGAR': 484,
 'RAJKOT': 485,
 'HAPUR': 486,
 'BHAVNAGAR': 487,
 'AGRA': 488,
 'JORHAT': 489,
 'MEDAK': 490,
 'FATEHABAD': 491,
 'BHARUCH': 492,
 'SONIPAT': 493,
 'RAMPUR': 494,
 'BONGAIGAON': 495,
 'PERAMBALUR': 496,
 '24 PARAGANAS NORTH': 497,
 'BALRAMPUR': 498,
 'NASHIK': 499,
 'SAMBHAL': 500,
 'GOALPARA': 501,
 'LATUR': 502,
 'JALGAON': 503,
 'AMBALA': 504,
 'ANANTAPUR': 505,
 'DHUBRI': 506,
 'DARRANG': 507,
 'BASTI': 508,
 'JUNAGADH': 509,
 'KARNAL': 510,
 'UDAM SINGH NAGAR': 511,
 'KAITHAL': 512,
 'JIND': 513,
 'SIRSA': 514,
 'GONDA': 515,
 'FARIDKOT': 516,
 'GUNTUR': 517,
 'BIRBHUM': 518,
 'COOCHBEHAR': 519,
 'BUDAUN': 520,
 'BEED': 521,
 'MANSA': 522,
 'PASHCHIM CHAMPARAN': 523,
 'HARDOI': 524,
 'HOSHIARPUR': 525,
 'NAWANSHAHR': 526,
 'BAGALKOT': 527,
 'MOGA': 528,
 'KARAIKAL': 529,
 'KURUKSHETRA': 530,
 'FATEHGARH SAHIB': 531,
 'BARNALA': 532,
 'SHAHJAHANPUR': 533,
 'BULANDSHAHR': 534,
 'MUKTSAR': 535,
 'KAPURTHALA': 536,
 'MARIGAON': 537,
 'YAMUNANAGAR': 538,
 'SATARA': 539,
 'KUSHI NAGAR': 540,
 'THIRUVALLUR': 541,
 'TARN TARAN': 542,
 'GHAZIABAD': 543,
 'BATHINDA': 544,
 'TUTICORIN': 545,
 'MORADABAD': 546,
 'PILIBHIT': 547,
 'SANGLI': 548,
 'MEDINIPUR WEST': 549,
 'SONITPUR': 550,
 'HARIDWAR': 551,
 'BAKSA': 552,
 'PATIALA': 553,
 'JALANDHAR': 554,
 'AHMEDNAGAR': 555,
 'NADIA': 556,
 'TIRUVANNAMALAI': 557,
 'FAZILKA': 558,
 'BARPETA': 559,
 'KARUR': 560,
 'BARDHAMAN': 561,
 'SPSR NELLORE': 562,
 'AMRITSAR': 563,
 'LUDHIANA': 564,
 'SITAPUR': 565,
 'SHAMLI': 566,
 'KHAMMAM': 567,
 'TIRUCHIRAPPALLI': 568,
 'BAREILLY': 569,
 'PUNE': 570,
 'AMROHA': 571,
 'BAGHPAT': 572,
 'NALBARI': 573,
 'MEDINIPUR EAST': 574,
 'GURDASPUR': 575,
 'HOWRAH': 576,
 'BELGAUM': 577,
 'SOLAPUR': 578,
 'HOOGHLY': 579,
 'SANGRUR': 580,
 'NAGAPATTINAM': 581,
 'GOLAGHAT': 582,
 'FIROZEPUR': 583,
 'MURSHIDABAD': 584,
 'KANCHIPURAM': 585,
 'KAMRUP': 586,
 'SURAT': 587,
 '24 PARAGANAS SOUTH': 588,
 'SAHARANPUR': 589,
 'YANAM': 590,
 'KHERI': 591,
 'CUDDALORE': 592,
 'KOLHAPUR': 593,
 'VILLUPURAM': 594,
 'PONDICHERRY': 595,
 'RAMANATHAPURAM': 596,
 'PUDUKKOTTAI': 597,
 'MEERUT': 598,
 'MAHE': 599,
 'DHARMAPURI': 600,
 'BIJNOR': 601,
 'SIVAGANGA': 602,
 'MADURAI': 603,
 'VIZIANAGARAM': 604,
 'NAMAKKAL': 605,
 'CHITTOOR': 606,
 'KRISHNA': 607,
 'MUZAFFARNAGAR': 608,
 'SALEM': 609,
 'VIRUDHUNAGAR': 610,
 'TIRUNELVELI': 611,
 'THIRUVARUR': 612,
 'NAGAON': 613,
 'ERODE': 614,
 'KANNIYAKUMARI': 615,
 'VELLORE': 616,
 'SOUTH ANDAMANS': 617,
 'NORTH AND MIDDLE ANDAMAN': 618,
 'VISAKHAPATANAM': 619,
 'KRISHNAGIRI': 620,
 'DINDIGUL': 621,
 'THENI': 622,
 'SRIKAKULAM': 623,
 'NORTH GOA': 624,
 'THANJAVUR': 625,
 'SOUTH GOA': 626,
 'WAYANAD': 627,
 'IDUKKI': 628,
 'TIRUPPUR': 629,
 'NICOBARS': 630,
 'COIMBATORE': 631,
 'WEST GODAVARI': 632,
 'PATHANAMTHITTA': 633,
 'EAST GODAVARI': 634,
 'KOTTAYAM': 635,
 'PALAKKAD': 636,
 'ERNAKULAM': 637,
 'KOLLAM': 638,
 'ALAPPUZHA': 639,
 'KASARAGOD': 640,
 'KANNUR': 641,
 'THIRUVANANTHAPURAM': 642,
 'THRISSUR': 643,
 'MALAPPURAM': 644,
 'KOZHIKODE': 645,
 'DADRA AND NAGARHAVELI' :136}


# In[6]:


Season_dict={'Summer': 0,
 'Autumn': 1,
 'Rabi': 2,
 'Kharif': 3,
 'Winter': 4,
 'Wholeyear': 5}


# In[7]:


Crop_dict={'Apple': 0,
 'Pump kin': 1,
 'Snak Guard': 2,
 'Cucumber': 3,
 'Lab-Lab': 4,
 'Plums': 5,
 'Ribed Guard': 6,
 'Litchi': 7,
 'Ber': 8,
 'Beet Root': 9,
 'Other Citrus Fruit': 10,
 'Pear': 11,
 'other fibres': 12,
 'Peas  (vegetable)': 13,
 'Yam': 14,
 'Peach': 15,
 'Ash Gourd': 16,
 'Water Melon': 17,
 'Bitter Gourd': 18,
 'Bottle Gourd': 19,
 'Turnip': 20,
 'Redish': 21,
 'Cond-spcs other': 22,
 'Jobster': 23,
 'Carrot': 24,
 'other misc. pulses': 25,
 'Perilla': 26,
 'Sannhamp': 27,
 'Cauliflower': 28,
 'Cashewnut Processed': 29,
 'Cardamom': 30,
 'Bean': 31,
 'Lentil': 32,
 'Cowpea(Lobia)': 33,
 'Ricebean (nagadal)': 34,
 'Blackgram': 35,
 'Linseed': 36,
 'Jack Fruit': 37,
 'Kapas': 38,
 'Niger seed': 39,
 'Drum Stick': 40,
 'Korra': 41,
 'Pome Granet': 42,
 'Varagu': 43,
 'Other Fresh Fruits': 44,
 'Bhindi': 45,
 'Rajmash Kholar': 46,
 'Horse-gram': 47,
 'Coriander': 48,
 'Sesamum': 49,
 'Small millets': 50,
 'Other Kharif pulses': 51,
 'Beans & Mutter(Vegetable)': 52,
 'Other  Rabi pulses': 53,
 'Moong(Green Gram)': 54,
 'Pome Fruit': 55,
 'Peas & beans (Pulses)': 56,
 'Cabbage': 57,
 'Sweet potato': 58,
 'Other Vegetables': 59,
 'Black pepper': 60,
 'Samai': 61,
 'Other Cereals & Millets': 62,
 'Citrus Fruit': 63,
 'Safflower': 64,
 'Sunflower': 65,
 'Cashewnut': 66,
 'Urad': 67,
 'Turmeric': 68,
 'Dry chillies': 69,
 'Tea': 70,
 'Garlic': 71,
 'Dry ginger': 72,
 'Masoor': 73,
 'Ginger': 74,
 'Cashewnut Raw': 75,
 'Brinjal': 76,
 'Moth': 77,
 'Tobacco': 78,
 'Castor seed': 79,
 'Khesari': 80,
 'Colocosia': 81,
 'Arhar/Tur': 82,
 'Barley': 83,
 'Jute & mesta': 84,
 'Mesta': 85,
 'Sapota': 86,
 'Lemon': 87,
 'Orange': 88,
 'Pineapple': 89,
 'Ragi': 90,
 'Papaya': 91,
 'other oilseeds': 92,
 'Arcanut (Processed)': 93,
 'Onion': 94,
 'Guar seed': 95,
 'Rapeseed &Mustard': 96,
 'Arecanut': 97,
 'Groundnut': 98,
 'Gram': 99,
 'Tomato': 100,
 'Grapes': 101,
 'Jowar': 102,
 'Maize': 103,
 'Bajra': 104,
 'Coffee': 105,
 'Mango': 106,
 'Rubber': 107,
 'Soyabean': 108,
 'Banana': 109,
 'Atcanut (Raw)': 110,
 'Potato': 111,
 'Pulses total': 112,
 'Paddy': 113,
 'Tapioca': 114,
 'Cotton(lint)': 115,
 'Oilseeds total': 116,
 'Rice': 117,
 'Jute': 118,
 'Wheat': 119,
 'Total foodgrain': 120,
 'Sugarcane': 121,
 'Coconut': 122}


# In[8]:


Year_dict={2015: 0,
 1997: 1,
 2001: 2,
 2010: 3,
 2007: 4,
 2006: 5,
 2002: 6,
 1999: 7,
 2009: 8,
 2008: 9,
 2003: 10,
 2000: 11,
 2012: 12,
 2004: 13,
 2005: 14,
 1998: 15,
 2013: 16,
 2014: 17,
 2011: 18,
 2016: 19,
 2017: 20,
 2018: 21,
 2019: 22,
 2020: 23,
 2021: 24,
 2022: 25,
 2023: 26,
 2024: 27,
 2025: 28,
 2026: 29,
 2027: 30,          
 2028: 31,         
 2029: 32,         
 2030: 33,         
 2031: 34,         
 2032: 35,         
 2033: 36,         
 2034: 37,         
 2035: 38,         
 2036: 39,         
 2037: 40,
 2038: 41,       
 2039: 42,       
 2040: 43,       
 2041: 44,       
 2042: 45,       
 2043: 46,       
 2044: 47,       
 2045: 48,
 2046: 49,          
 2047: 50,          
 2048: 51,          
 2049: 52,          
 2050: 53,          
 2051: 54}
       
# In[9]:


a="Andaman & Nicobar Island"


# In[10]:


print(State_dict.get(a))


# In[11]:

import datetime
now = datetime.datetime.now()
check_year=now.year
print(check_year)

predict_year=check_year+2
print('predict_year',predict_year)
standard_to = StandardScaler()
@app.route('/predict',methods=['POST'])
def predict():
    #int_features=[String(x) for x in request.form.values()]
    #final_features =[np.array(int_features)]
    if request.method == 'POST':
        State_Name=request.form['State_Name']
        State_Name1=State_Name
        State_Name=State_Name.capitalize()
        State_Name=State_dict.get(State_Name)
        District_Name  =request.form['District_Name']
        District_Name1=District_Name
        District_Name=District_Name.upper()
        District_Name=Dist_dict.get(District_Name)
        Crop_Year=int(request.form['Crop_Year'])
        Crop_Year_original=Crop_Year
        if Crop_Year<1998:
            Crop_Year1=Crop_Year
            #Crop_Year=1
            Crop_Year2=Crop_Year-2
            Crop_Year7=Crop_Year2
            Crop_Year2=1
            Crop_Year3=Crop_Year-1
            Crop_Year8=Crop_Year3
            Crop_Year3=1
            Crop_Year4=Crop_Year+1
            Crop_Year9=Crop_Year4
            Crop_Year4=1
            Crop_Year5=Crop_Year+2
            Crop_Year10=Crop_Year5
            Crop_Year5=1
            Crop_Year=1
        else:
            Crop_Year1=Crop_Year
            Crop_Year2=Crop_Year-2
            Crop_Year7=Crop_Year2
            Crop_Year3=Crop_Year-1
            Crop_Year8=Crop_Year3
            Crop_Year4=Crop_Year+1
            Crop_Year9=Crop_Year4
            Crop_Year5=Crop_Year+2
            Crop_Year10=Crop_Year5
            if Crop_Year<2052:
                Crop_Year=Year_dict.get(Crop_Year)
            else:
                Crop_Year=55
            if Crop_Year2<2052:
                Crop_Year2=Year_dict.get(Crop_Year2)
            else:
                Crop_Year2=55
            if Crop_Year3<2052:
                Crop_Year3=Year_dict.get(Crop_Year3)
            else:
                Crop_Year3=55
            if Crop_Year4<2052:
                Crop_Year4=Year_dict.get(Crop_Year4)
            else:
                Crop_Year4=55
            if Crop_Year5<2052:
                Crop_Year5=Year_dict.get(Crop_Year5)
            else:
                Crop_Year5=55
        Season=request.form['Season']
        Season1=Season
        Season=Season.capitalize()
        Season=Season_dict.get(Season)
        Crop=request.form['Crop']
        #Crop=Crop.capitalize()
        Crop1=Crop
        if Crop1=='Arhar/Tur':
            crop2='ArharTur'
        elif Crop1=='Castor seed':
            crop2='Castorseed'
        elif Crop1=='Cond-spcs other':
            crop2='Condspcsother'
        elif Crop1=='Cotton(lint)':
            crop2='Cottonlint'
        elif Crop1=='Moong(Green Gram)':
            crop2='MoongGreenGram'
        elif Crop1=='Rapeseed &Mustard':
            crop2='RapeseedMustard'
        else:
            crop2=Crop1
        Crop=Crop_dict.get(Crop)
        Area=float(request.form['Area'])
        Area1=Area

    print([[State_Name,District_Name,Crop_Year,Season,Crop,Area]] ) 
    print([[State_Name,District_Name,Crop_Year2,Season,Crop,Area]] )
    print([[State_Name,District_Name,Crop_Year3,Season,Crop,Area]] )
    print([[State_Name,District_Name,Crop_Year4,Season,Crop,Area]] )
    print([[State_Name,District_Name,Crop_Year5,Season,Crop,Area]] )
    if Area<10:
        Area1=Area+5
        Area2=Area1+5
        Area3=Area2+5
        Area4=Area3+5
    elif Area<100:
        Area1=Area+10
        Area2=Area1+10
        Area3=Area2+10
        Area4=Area3+10
    elif Area<1000:
        Area1=Area+50
        Area2=Area1+50
        Area3=Area2+50
        Area4=Area3+50
    elif Area<10000:
        Area1=Area+100
        Area2=Area1+100
        Area3=Area2+100
        Area4=Area3+100
    elif Area>9999:
        Area1=Area+1000
        Area2=Area1+1000
        Area3=Area2+1000
        Area4=Area3+1000
    
    prediction=model.predict([[State_Name,District_Name,Crop_Year,Season,Crop,Area]] ) 
    prediction1=model.predict([[State_Name,District_Name,Crop_Year2,Season,Crop,Area]] )
    prediction2=model.predict([[State_Name,District_Name,Crop_Year3,Season,Crop,Area]] )
    prediction3=model.predict([[State_Name,District_Name,Crop_Year4,Season,Crop,Area]] )
    prediction4=model.predict([[State_Name,District_Name,Crop_Year5,Season,Crop,Area]] )
    prediction_Area1=model.predict([[State_Name,District_Name,Crop_Year,Season,Crop,Area1]] )
    prediction_Area2=model.predict([[State_Name,District_Name,Crop_Year,Season,Crop,Area2]] )
    prediction_Area3=model.predict([[State_Name,District_Name,Crop_Year,Season,Crop,Area3]] )
    prediction_Area4=model.predict([[State_Name,District_Name,Crop_Year,Season,Crop,Area4]] )
        
    
    
    print('Crop_Year',Crop_Year_original)
    if predict_year < Crop_Year_original:
        prediction=0
        print('year_if')
        return render_template('index.html',
                                   prediction_text='This tool is not predict for future',
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Arhar/Tur
    if State_Name==30 and Crop==82 or State_Name==10 and Crop==82 or State_Name==6 and Crop==82 or State_Name==3 and Crop==82 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Banana
    if State_Name==21 and Crop==109 or State_Name==8 and Crop==109 or State_Name==10 and Crop==109 or State_Name==26 and Crop==109 or State_Name==7 and Crop==109 or State_Name==17 and Crop==109 or State_Name==0 and Crop==109 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    
    #Cabbage & Cauliflower
    
    if State_Name==30 and Crop==57 or State_Name==30 and Crop==28 or State_Name==20 and Crop==28 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Groundnut
    if State_Name==24 and Crop==98 or State_Name==10 and Crop==98 or State_Name==1 and Crop==98 or State_Name==5 and Crop==98 or State_Name==3 and Crop==98 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Maize
    if State_Name==30 and Crop==103 or State_Name==32 and Crop==103 or State_Name==27 and Crop==103 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1)) 
    #Mango
    if State_Name==5 and Crop==106 or State_Name==6 and Crop==106 or State_Name==3 and Crop==106 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    
     #Moong
    if State_Name==5 and Crop==54 or State_Name==6 and Crop==54 or State_Name==3 and Crop==54 or State_Name==30 and Crop==54 or State_Name==1 and Crop==54 or State_Name==4 and Crop==54:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Onion
    if State_Name==2 and Crop==94 or State_Name==30 and Crop==94 or State_Name==32 and Crop==94 or State_Name==3 and Crop==94 or State_Name==31 and Crop==94 or State_Name==7 and Crop==94:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Potato & Rapseed(96)
    if State_Name==30 and Crop==111 or State_Name==32 and Crop==111 or State_Name==31 and Crop==111 or State_Name==30 and Crop==96 or State_Name==32 and Crop==96:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Soyabean
    if State_Name==30 and Crop==108 or State_Name==24 and Crop==108 or State_Name==15 and Crop==108 or State_Name==21 and Crop==108 or State_Name==10 and Crop==108 or State_Name==32 and Crop==108 or State_Name==26 and Crop==108 or State_Name==9 and Crop==108 or State_Name==28 and Crop==108:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Sugarcane & Urad
    if State_Name==3 and Crop==121 or State_Name==0 and Crop==121 or State_Name==30 and Crop==67 or State_Name==6 and Crop==67 or State_Name==5 and Crop==67 or State_Name==3 and Crop==67 or State_Name==32 and Crop==67 or State_Name==1 and Crop==67:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Wheat
    
    if State_Name==29 and Crop==119 or State_Name==32 and Crop==119 or State_Name==1 and Crop==119 or State_Name==30 and Crop==119 or State_Name==27 and Crop==119 or State_Name==31 and Crop==119:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Rice
    if State_Name==10 and Crop==117 or State_Name==16 and Crop==117 or State_Name==5 and Crop==117 or State_Name==9 and Crop==117 or State_Name==1 and Crop==117 or State_Name==6 and Crop==117 or State_Name==2 and Crop==117:
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    
    #Barley
    
    if State_Name==0 and Crop==83 or State_Name==1 and Crop==83 or State_Name==2 and Crop==83 or State_Name==4 and Crop==83 or State_Name==5 and Crop==83 or State_Name==6 and Crop==83 or State_Name==7 and Crop==83 or State_Name==9 and Crop==83 or State_Name==10 and Crop==83 or State_Name==13 and Crop==83 or State_Name==18 and Crop==83 or State_Name==19 and Crop==83 or State_Name==20 and Crop==83 or State_Name==22 and Crop==83 or State_Name==24 and Crop==83 or State_Name==27 and Crop==83 or State_Name==28 and Crop==83 or State_Name==29 and Crop==83 or State_Name==30 and Crop==83 or State_Name==31 and Crop==83 or State_Name==32 and Crop==83 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Cashewnut
    
    if State_Name==0 and Crop==66 or State_Name==1 and Crop==66 or State_Name==2 and Crop==66 or State_Name==3 and Crop==66 or State_Name==6 and Crop==66 or State_Name==7 and Crop==66 or State_Name==8 and Crop==66 or State_Name==11 and Crop==66 or State_Name==13 and Crop==66 or State_Name==14 and Crop==66 or State_Name==15 and Crop==66 or State_Name==16 and Crop==66 or State_Name==17 and Crop==66 or State_Name==21 and Crop==66 or State_Name==23 and Crop==66 or State_Name==26 and Crop==66 or State_Name==31 and Crop==66 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #castor
    
    if State_Name==0 and Crop==79 or State_Name==1 and Crop==79 or State_Name==2 and Crop==79 or State_Name==3 and Crop==79 or State_Name==5 and Crop==79 or State_Name==6 and Crop==79 or State_Name==7 and Crop==79 or State_Name==8 and Crop==79 or State_Name==9 and Crop==79 or State_Name==10 and Crop==79 or State_Name==11 and Crop==79 or State_Name==12 and Crop==79 or State_Name==17 and Crop==79 or State_Name==23 and Crop==79 or State_Name==25 and Crop==79 or State_Name==26 and Crop==79 or State_Name==27 and Crop==79 or State_Name==30 and Crop==79 or State_Name==31 and Crop==79 or State_Name==32 and Crop==79 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
     #coconut
    
    if State_Name==0 and Crop==122 or State_Name==1 and Crop==122 or State_Name==2 and Crop==122 or State_Name==3 and Crop==122 or State_Name==5 and Crop==122 or State_Name==6 and Crop==122 or State_Name==7 and Crop==122 or State_Name==8 and Crop==122 or State_Name==10 and Crop==122 or State_Name==11 and Crop==122 or State_Name==12 and Crop==122 or State_Name==14 and Crop==122 or State_Name==15 and Crop==122 or State_Name==16 and Crop==122 or State_Name==17 and Crop==122 or State_Name==21 and Crop==122 or State_Name==23 and Crop==122 or State_Name==26 and Crop==122 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #Coffee
    
    if State_Name==0 and Crop==105 or State_Name==1 and Crop==105 or State_Name==2 and Crop==105 or State_Name==3 and Crop==105 or State_Name==4 and Crop==105 or State_Name==5 and Crop==105 or State_Name==6 and Crop==105 or State_Name==7 and Crop==105 or State_Name==8 and Crop==105 or State_Name==9 and Crop==105 or State_Name==10 and Crop==105 or State_Name==11 and Crop==105 or State_Name==12 and Crop==105 or State_Name==14 and Crop==105 or State_Name==15 and Crop==105 or State_Name==16 and Crop==105 or State_Name==17 and Crop==105 or State_Name==19 and Crop==105 or State_Name==20 and Crop==105 or State_Name==21 and Crop==105 or State_Name==22 and Crop==105 or State_Name==23 and Crop==105 or State_Name==25 and Crop==105 or State_Name==26 and Crop==105 or State_Name==27 and Crop==105 or State_Name==30 and Crop==105 or State_Name==31 and Crop==105 :
        
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
     #CONDIMENTS 
    
    if State_Name==0 and Crop==22 or State_Name==1 and Crop==22 or State_Name==2 and Crop==22 or State_Name==3 and Crop==22 or State_Name==4 and Crop==22 or State_Name==5 and Crop==22 or State_Name==7 and Crop==22 or State_Name==8 and Crop==22 or State_Name==9 and Crop==22 or State_Name==11 and Crop==22 or State_Name==13 and Crop==22 or State_Name==15 and Crop==22 or State_Name==27 and Crop==22 or State_Name==30 and Crop==22 or State_Name==31 and Crop==22 or State_Name==32 and Crop==22 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
     #Cotton iint 
    
    if State_Name==0 and Crop==115 or State_Name==1 and Crop==115 or State_Name==2 and Crop==115 or State_Name==3 and Crop==115 or State_Name==4 and Crop==115 or State_Name==6 and Crop==115 or State_Name==7 and Crop==115 or State_Name==8 and Crop==115 or State_Name==9 and Crop==115 or State_Name==10 and Crop==115 or State_Name==11 and Crop==115 or State_Name==12 and Crop==115 or State_Name==15 and Crop==115 or State_Name==17 and Crop==115 or State_Name==25 and Crop==115 or State_Name==27 and Crop==115 or State_Name==30 and Crop==115 or State_Name==31 and Crop==115 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    #jowar
    
    if State_Name==0 and Crop==102 or State_Name==1 and Crop==102 or State_Name==2 and Crop==102 or State_Name==3 and Crop==102 or State_Name==4 and Crop==102 or State_Name==5 and Crop==102 or State_Name==6 and Crop==102 or State_Name==7 and Crop==102 or State_Name==8 and Crop==102 or State_Name==9 and Crop==102 or State_Name==11 and Crop==102 or State_Name==13 and Crop==102 or State_Name==17 and Crop==102 or State_Name==24 and Crop==102 or State_Name==25 and Crop==102 or State_Name==26 and Crop==102 or State_Name==27 and Crop==102 or State_Name==30 and Crop==102 or State_Name==31 and Crop==102 or State_Name==32 and Crop==102 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
     #sunflower
    
    if State_Name==0 and Crop==65 or State_Name==1 and Crop==65 or State_Name==3 and Crop==65 or State_Name==5 and Crop==65 or State_Name==6 and Crop==65 or State_Name==7 and Crop==65 or State_Name==8 and Crop==65 or State_Name==9 and Crop==65 or State_Name==11 and Crop==65 or State_Name==14 and Crop==65 or State_Name==16 and Crop==65 or State_Name==17 and Crop==65 or State_Name==19 and Crop==65 or State_Name==24 and Crop==65 or State_Name==27 and Crop==65 or State_Name==30 and Crop==65 or State_Name==31 and Crop==65  or State_Name==32 and Crop==65 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
     #Tea
    
    if State_Name==0 and Crop==70 or State_Name==1 and Crop==70 or State_Name==3 and Crop==70 or State_Name==5 and Crop==70 or State_Name==6 and Crop==70 or State_Name==7 and Crop==70 or State_Name==8 and Crop==70 or State_Name==11 and Crop==70 or State_Name==12 and Crop==70 or State_Name==13 and Crop==70 or State_Name==14 and Crop==70 or State_Name==16 and Crop==70 or State_Name==17 and Crop==70 or State_Name==19 and Crop==70 or State_Name==20 and Crop==70 or State_Name==21 and Crop==70 or State_Name==22 and Crop==70 or State_Name==23 and Crop==70 or State_Name==26 and Crop==70 or State_Name==27 and Crop==70 or State_Name==29 and Crop==70 or State_Name==30 and Crop==70 or State_Name==31 and Crop==70 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
    
    
    #Tobacco
    
    if State_Name==0 and Crop==78 or State_Name==2 and Crop==78 or State_Name==3 and Crop==78 or State_Name==4 and Crop==78 or State_Name==6 and Crop==78 or State_Name==7 and Crop==78 or State_Name==8 and Crop==78 or State_Name==9 and Crop==78 or State_Name==11 and Crop==78 or State_Name==17 and Crop==78 or State_Name==21 and Crop==78 or State_Name==26 and Crop==78 or State_Name==27 and Crop==78 or State_Name==29 and Crop==78 or State_Name==30 and Crop==78 or State_Name==31 and Crop==78 :
        prediction=0
        return render_template('index.html',
                                   prediction_text='crop production prediction {} tonne because this crop is not grown in this state'.format(prediction),
                                   state='State_Name :{}'.format(State_Name1),
                                   district='District_Name :{}'.format(District_Name1),
                                   year='Crop_Year :{}'.format(Crop_Year1),
                                   season='Season :{}'.format(Season1),
                                   crop='Crop :{}'.format(Crop1),
                                   Area='Area :{} ha'.format(Area1))
        
            
       
            
        
            
    else:
        return render_template('index.html',prediction_text='crop production prediction {} tonne'.format(prediction),
                              state='State_Name :{}'.format(State_Name1),
                              district='District_Name :{}'.format(District_Name1),
                              year='Crop_Year :{}'.format(Crop_Year1),
                              season='Season :{}'.format(Season1),
                              crop='Crop :{}'.format(Crop1),
                              crop_div='Description of Crop :{}'.format(Crop1),
                              crop_out=crop2,
                              Area='Area :{} ha'.format(Area1),
                              #prediction_text1='crop production prediction1 {} tonne'.format(prediction1),
                              #prediction_text2='crop production prediction2 {} tonne'.format(prediction2),
                              #prediction_text3='crop production prediction3 {} tonne'.format(prediction3),
                              #prediction_text4='crop production prediction4 {} tonne'.format(prediction4),
                              prediction_text1=prediction1,
                              prediction_text2=prediction2,
                              prediction_text3=prediction,
                              prediction_text4=prediction3,
                              prediction_text5=prediction4,
                              prediction_Area=prediction,
                              prediction_Area1=prediction_Area1,
                              prediction_Area2=prediction_Area2,
                              prediction_Area3=prediction_Area3,
                              prediction_Area4=prediction_Area4,
                              
                              Area_enter=Area,
                              Area1=Area1,
                              Area2=Area2,
                              Area3=Area3,
                              Area4=Area4,
                              crop1=Crop_Year7,
                              crop2=Crop_Year8,
                              crop3=Crop_Year1,
                              crop4=Crop_Year9,
                              crop5=Crop_Year10)
                              #crop2='Crop_Year :{}'.format(Crop_Year8),
                              #crop3='Crop_Year :{}'.format(Crop_Year9),
                              #crop4='Crop_Year :{}'.format(Crop_Year10))


# In[12]:


if __name__=="__main__":
    app.run(debug=True)


# In[ ]:




