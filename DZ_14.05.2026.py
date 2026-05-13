import kagglehub
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, Normalizer

path = kagglehub.dataset_download("saurabhshahane/ecommerce-text-classification")

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svd", TruncatedSVD(n_components=15)),
    ("scaler", StandardScaler()),
    ("normalizer", Normalizer()),
    ("clf", LogisticRegression()),
])
 
texts = [
    #household
    "Paper Plane Design Framed Wall Hanging Motivational Office Decor Art Prints (8.7 X 8.7 inch) - Set of 4 Painting made up in synthetic frame with uv textured print which gives multi effects and attracts towards it.",
    "SAF 'Floral' Framed Painting (Wood, 30 inch x 10 inch, Special Effect UV Print Textured, SAO297) Painting made up in synthetic frame with UV textured print which gives multi effects and attracts towards it.",
    "SAF 'UV Textured Modern Art Print Framed' Painting (Synthetic, 35 cm x 50 cm x 3 cm, Set of 3) Color:Multicolor Overview a beautiful painting involves the action or skill of using paint in the right manner.",
    "SAF Flower Print Framed Painting (Synthetic, 13.5 inch x 22 inch, UV Textured, Set of 3, SANFSW4951) A beautiful painting involves the action or skill of using paint in the right manner, hence, the end product will be a picture.",
    "Incredible Gifts India Wooden Happy Birthday Unique Personalized Gift (5 X 4 Inch) Made Of Natural Imported Wood, Which Is Quite Solid With Light Particle Pattern & Is Soft Pale To Blond Colour.",
    "Pitaara Box Romantic Venice Canvas Painting 6mm Thick Mdf Frame 21.1 X 14Inch Enhance the beauty of your room walls with this breathtaking digital printed artwork.",
    
    #books
    "UPSC New Syllabus & Tips to Crack IAS Preliminary and Mains Exam with Rapid GK 2019 ebook About the Author Disha Experts is a team of most renowned and prolific content writers pioneering in School and Test Prep segments.",
    "IAS General Studies Preliminary Topic wise Solved Papers (Paper I and II) The book includes Previous Years Solved Papers of IAS General Studies (Preliminary) Examination conducted by UPSC.",
    "ESSAYS for Civil Services and Other Competitive Examinations About the Author The Author is IAS Topper (Rank 5, Civil Services Examination: 2010). His interests include reading, swimming and playing synthesizer.",
    "24 Years UPSC IAS/ IPS Prelims Topic-wise Solved Papers 1 & 2 (1995-2018)",
    "UPSC Civil Services 2011-2018 (Prelim) GS & CSAT Yearwise & Topicwise Solved Papers English - 2339",
    "General Studies Manual Paper - 1 2019 About the Author An editorial team of highly skilled professionals at Arihant, works hand in glove to ensure that the students receive the best and accurate content through our books.",
    
    #clothing and accessories
    "VIMAL Men's Black Cotton Trackpants Vimal trackpants made from 100% rich combed cotton, giving it a rich look, Skin friendly fabric for comfort of wearing, itch-free waistband.",
    "Van Heusen Men's Cotton Rich Jogger Pant Everyday comfort meets fashion in this track pant from Van Heusen. Made from rich cotton with stretch, these pants are fitted with an elasticated waistband.",
    "Chromozome Men's Cotton Track Pants",
    "4 Way Lycra Stylish Navy Trackpant for Men with Two Side Zipper Pockets – Stretchable, Comfortable & Absorbent Slim Fit Track Pants for Gym Workout and Casual Wear.",
    "Van Heusen Men's Cotton Rich Track Pant Redefine comfort with these exquisitely styled fashion shorts from Van Heusen. Crafted in cotton rich fabric, these athleisure shorts with drawstring fastening.",
    
    #electronics
    "Brain Freezer J DSLR SLR Camera Lens Shoulder Backpack Case for Canon Nikon Sigma Olympus (Black)",
    "Aputure AL-M9 Amaran LED Mini Light on Camera Video Light, Black Introduction: The Amaran AL-M9 is a pocket sized LED fill light. It is incredibly compact and lightweight.",
    "Godox CB-09 Hard Carrying Storage Suitcase Carry Bag For AD600 AD600B AD600BM AD360 TT685 Flash Kit Designed for studio flash and accessories.",
    "Sonia Combo of Umbrella Flash Light Stand for Photography Set of 2 with Carry Bag Case It is widely used in small to large photo studio. Beginner to professional, all photographer.",
    "Amazing World of Needs Light Stand Carry Bag/Case Light stand is constructed from aluminum alloy, giving it exceptional strength for heavy work."
]
 
y = [
    "Household", "Household", "Household", "Household", "Household", "Household",
    "Books", "Books", "Books", "Books", "Books", "Books",
    "Clothing & Accessories", "Clothing & Accessories", "Clothing & Accessories", "Clothing & Accessories", "Clothing & Accessories",
    "Electronics", "Electronics", "Electronics", "Electronics", "Electronics"
]
 
model.fit(texts, y)
 
print(model.predict(["Beautiful wall painting for living room decor"]))
print(model.predict(["Wireless Bluetooth gaming mouse with RGB light"]))
print(model.predict(["Data science handbook and python coding textbook for students"]))
print(model.predict(["Wooden photo frame personalized for anniversary gift"]))
print(model.predict(["4K Ultra HD Smart TV with HDMI and Wi-Fi connection"]))
print(model.predict(["Fantasy novel hardcover book with maps"]))
print(model.predict(["Abstract oil painting on canvas with wooden frame for home decoration"]))
print(model.predict(["UPSC civil services exam preparation textbook and solved papers guide"]))
print(model.predict(["Men's slim fit sports joggers track pants 100% cotton casual wear"]))
print(model.predict(["DSLR camera travel backpack with shoulder strap and tripod holder"]))
print(model.predict(["Beautiful decorative flower print for office wall hanging"]))
print(model.predict(["Lycra gym clothing for workout and running"]))

'''
output:

['Household']
['Electronics']
['Books']
['Electronics']
['Books']
['Books']
['Household']
['Books']
['Clothing & Accessories']
['Electronics']
['Household']
['Clothing & Accessories']
'''
