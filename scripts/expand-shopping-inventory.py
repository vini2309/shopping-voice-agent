from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "backend" / "app" / "data" / "inventory.json"


PRIVATE_LABEL = "Everyday Choice"

BRAND_OVERRIDES = {
    "Bounty": "Bounty",
    "Clorox": "Clorox",
    "Dawn": "Dawn",
    "Tide": "Tide",
    "Downy": "Downy",
    "Hefty": "Hefty",
    "Ziploc": "Ziploc",
    "Reynolds": "Reynolds",
    "Glad": "Glad",
    "Eggland": "Eggland's Best",
    "Land O Lakes": "Land O Lakes",
    "Sargento": "Sargento",
    "Yoplait": "Yoplait",
    "Thomas": "Thomas",
    "Jif": "Jif",
    "Smucker": "Smucker's",
    "Barilla": "Barilla",
    "Prego": "Prego",
    "Kellogg": "Kellogg's",
    "Quaker": "Quaker",
    "Folgers": "Folgers",
    "Lipton": "Lipton",
    "Coca": "Coca-Cola",
    "Lay": "Lay's",
    "Doritos": "Doritos",
    "Oreo": "Oreo",
    "Gold Medal": "Gold Medal",
    "McCormick": "McCormick",
    "Tyson": "Tyson",
    "Market Fresh": "Market Fresh",
    "Oscar Mayer": "Oscar Mayer",
    "Dole": "Dole",
    "DiGiorno": "DiGiorno",
    "Ben & Jerry": "Ben & Jerry's",
    "Huggies": "Huggies",
    "Pampers": "Pampers",
    "Colgate": "Colgate",
    "Oral-B": "Oral-B",
    "Tylenol": "Tylenol",
    "Equate": "Equate",
    "Purina": "Purina",
}

DEPARTMENT_META = {
    "Household Paper": ("Household", "Paper Goods"),
    "Cleaning": ("Household", "Cleaning"),
    "Laundry": ("Household", "Laundry"),
    "Household Essentials": ("Household", "Trash and Storage"),
    "Food Storage": ("Household", "Food Storage"),
    "Dairy": ("Grocery", "Dairy"),
    "Bakery": ("Grocery", "Bakery"),
    "Pantry": ("Grocery", "Pantry"),
    "Breakfast": ("Grocery", "Breakfast"),
    "Beverages": ("Grocery", "Beverages"),
    "Snacks": ("Grocery", "Snacks"),
    "Baking": ("Grocery", "Baking"),
    "Meat": ("Grocery", "Meat"),
    "Deli": ("Grocery", "Deli"),
    "Produce": ("Grocery", "Produce"),
    "Frozen": ("Grocery", "Frozen"),
    "Baby": ("Baby", "Baby Care"),
    "Health": ("Health", "Personal Care"),
    "Pharmacy": ("Health", "Medicine"),
    "Pets": ("Pets", "Pet Food"),
}

ADDITIONAL_PRODUCTS = [
    {"name": "Everyday Choice AA Alkaline Batteries 24 Count", "brand": PRIVATE_LABEL, "department": "Electronics", "category": "Electronics", "subcategory": "Batteries", "aisle": "E01", "bay": "02", "stock": 34, "price": 12.49, "notes": "Battery wall near flashlights.", "synonyms": ["aa batteries", "batteries", "alkaline batteries"], "size": "24 count", "unit": "pack"},
    {"name": "PowerLine USB-C Fast Charging Cable 6 ft", "brand": "PowerLine", "department": "Electronics", "category": "Electronics", "subcategory": "Phone Accessories", "aisle": "E02", "bay": "04", "stock": 27, "price": 9.99, "notes": "Hanging peg hooks by phone chargers.", "synonyms": ["usb c cable", "charging cable", "phone cable"], "size": "6 ft", "unit": "each"},
    {"name": "PowerLine 20W USB-C Wall Charger", "brand": "PowerLine", "department": "Electronics", "category": "Electronics", "subcategory": "Phone Accessories", "aisle": "E02", "bay": "05", "stock": 18, "price": 14.99, "notes": "Small box chargers next to cables.", "synonyms": ["wall charger", "usb c charger", "phone charger"], "size": "20 watt", "unit": "each"},
    {"name": "ClearView HDMI Cable 6 ft", "brand": "ClearView", "department": "Electronics", "category": "Electronics", "subcategory": "TV Accessories", "aisle": "E03", "bay": "01", "stock": 16, "price": 8.98, "notes": "TV accessory shelf below remotes.", "synonyms": ["hdmi cable", "tv cable", "monitor cable"], "size": "6 ft", "unit": "each"},
    {"name": "BrightHome LED Light Bulbs Soft White 8 Pack", "brand": "BrightHome", "department": "Home Improvement", "category": "Home", "subcategory": "Lighting", "aisle": "H15", "bay": "03", "stock": 21, "price": 10.98, "notes": "Lighting aisle, soft white shelf.", "synonyms": ["light bulbs", "led bulbs", "soft white bulbs"], "size": "8 pack", "unit": "box"},
    {"name": "SafeStrip 6 Outlet Surge Protector", "brand": "SafeStrip", "department": "Electronics", "category": "Electronics", "subcategory": "Power", "aisle": "E03", "bay": "06", "stock": 12, "price": 16.97, "notes": "Power strip section near extension cords.", "synonyms": ["surge protector", "power strip", "extension strip"], "size": "6 outlet", "unit": "each"},
    {"name": "ClickPro Wireless Mouse", "brand": "ClickPro", "department": "Electronics", "category": "Electronics", "subcategory": "Computer Accessories", "aisle": "E04", "bay": "02", "stock": 9, "price": 13.88, "notes": "Computer accessory peg wall.", "synonyms": ["wireless mouse", "computer mouse", "mouse"], "size": "standard", "unit": "each"},
    {"name": "SoundPod Mini Bluetooth Speaker", "brand": "SoundPod", "department": "Electronics", "category": "Electronics", "subcategory": "Audio", "aisle": "E05", "bay": "03", "stock": 8, "price": 24.96, "notes": "Portable audio case near earbuds.", "synonyms": ["bluetooth speaker", "speaker", "portable speaker"], "size": "mini", "unit": "each"},
    {"name": "Everyday Choice Notebook Wide Ruled 1 Subject", "brand": PRIVATE_LABEL, "department": "Office", "category": "Office", "subcategory": "School Supplies", "aisle": "O01", "bay": "01", "stock": 80, "price": 1.24, "notes": "Back-to-school supply shelf.", "synonyms": ["notebook", "school notebook", "wide ruled notebook"], "size": "1 subject", "unit": "each"},
    {"name": "WriteRight Ballpoint Pens Black 10 Count", "brand": "WriteRight", "department": "Office", "category": "Office", "subcategory": "Writing", "aisle": "O01", "bay": "04", "stock": 42, "price": 2.48, "notes": "Pen hooks beside pencils.", "synonyms": ["pens", "black pens", "ballpoint pens"], "size": "10 count", "unit": "pack"},
    {"name": "Everyday Choice Copy Paper 500 Sheets", "brand": PRIVATE_LABEL, "department": "Office", "category": "Office", "subcategory": "Paper", "aisle": "O02", "bay": "01", "stock": 28, "price": 5.98, "notes": "Bottom shelf with printer paper reams.", "synonyms": ["copy paper", "printer paper", "paper ream"], "size": "500 sheets", "unit": "ream"},
    {"name": "StickEase Transparent Tape 6 Rolls", "brand": "StickEase", "department": "Office", "category": "Office", "subcategory": "Tape", "aisle": "O02", "bay": "05", "stock": 17, "price": 4.76, "notes": "Tape shelf by scissors and glue.", "synonyms": ["tape", "transparent tape", "office tape"], "size": "6 rolls", "unit": "pack"},
    {"name": "Everyday Choice Bath Towels Gray 4 Pack", "brand": PRIVATE_LABEL, "department": "Home", "category": "Home", "subcategory": "Bath Linens", "aisle": "L01", "bay": "02", "stock": 13, "price": 18.98, "notes": "Folded towel wall, gray color stack.", "synonyms": ["bath towels", "towels", "gray towels"], "size": "4 pack", "unit": "set"},
    {"name": "SleepNest Standard Bed Pillows 2 Pack", "brand": "SleepNest", "department": "Home", "category": "Home", "subcategory": "Bedding", "aisle": "L02", "bay": "01", "stock": 19, "price": 14.96, "notes": "Bedding aisle top shelf.", "synonyms": ["pillows", "bed pillows", "standard pillows"], "size": "2 pack", "unit": "set"},
    {"name": "SleepNest Queen Sheet Set White", "brand": "SleepNest", "department": "Home", "category": "Home", "subcategory": "Bedding", "aisle": "L02", "bay": "04", "stock": 10, "price": 24.98, "notes": "Sheet sets arranged by bed size.", "synonyms": ["queen sheets", "sheet set", "white sheets"], "size": "queen", "unit": "set"},
    {"name": "ClearBox Storage Tote 18 Gallon", "brand": "ClearBox", "department": "Home", "category": "Home", "subcategory": "Storage", "aisle": "L04", "bay": "Pallet 1", "stock": 32, "price": 9.98, "notes": "Stacked totes in home organization.", "synonyms": ["storage tote", "storage bin", "plastic tote"], "size": "18 gallon", "unit": "each"},
    {"name": "Everyday Choice Laundry Basket White", "brand": PRIVATE_LABEL, "department": "Home", "category": "Home", "subcategory": "Laundry Storage", "aisle": "L04", "bay": "06", "stock": 15, "price": 6.44, "notes": "Nested baskets near hampers.", "synonyms": ["laundry basket", "basket", "clothes basket"], "size": "standard", "unit": "each"},
    {"name": "CookWell Nonstick Frying Pan 10 Inch", "brand": "CookWell", "department": "Kitchen", "category": "Home", "subcategory": "Cookware", "aisle": "K01", "bay": "03", "stock": 11, "price": 15.97, "notes": "Cookware wall by skillets.", "synonyms": ["frying pan", "skillet", "nonstick pan"], "size": "10 inch", "unit": "each"},
    {"name": "CookWell Plastic Cutting Board Set 3 Pack", "brand": "CookWell", "department": "Kitchen", "category": "Home", "subcategory": "Kitchen Tools", "aisle": "K02", "bay": "05", "stock": 14, "price": 8.97, "notes": "Cutting board rack near knives.", "synonyms": ["cutting board", "cutting boards", "kitchen board"], "size": "3 pack", "unit": "set"},
    {"name": "Everyday Choice Paper Plates 100 Count", "brand": PRIVATE_LABEL, "department": "Party", "category": "Household", "subcategory": "Disposable Tableware", "aisle": "A10", "bay": "01", "stock": 30, "price": 5.48, "notes": "Disposable tableware section.", "synonyms": ["paper plates", "plates", "disposable plates"], "size": "100 count", "unit": "pack"},
    {"name": "Everyday Choice Plastic Cups 120 Count", "brand": PRIVATE_LABEL, "department": "Party", "category": "Household", "subcategory": "Disposable Tableware", "aisle": "A10", "bay": "02", "stock": 26, "price": 4.98, "notes": "Plastic cups beside paper plates.", "synonyms": ["plastic cups", "cups", "party cups"], "size": "120 count", "unit": "pack"},
    {"name": "PureCare Shampoo Hydrating 28 fl oz", "brand": "PureCare", "department": "Beauty", "category": "Beauty", "subcategory": "Hair Care", "aisle": "HB01", "bay": "02", "stock": 22, "price": 6.98, "notes": "Hair care aisle, hydrating shampoo shelf.", "synonyms": ["shampoo", "hydrating shampoo", "hair shampoo"], "size": "28 fl oz", "unit": "bottle"},
    {"name": "PureCare Conditioner Hydrating 28 fl oz", "brand": "PureCare", "department": "Beauty", "category": "Beauty", "subcategory": "Hair Care", "aisle": "HB01", "bay": "03", "stock": 20, "price": 6.98, "notes": "Beside matching shampoo.", "synonyms": ["conditioner", "hair conditioner", "hydrating conditioner"], "size": "28 fl oz", "unit": "bottle"},
    {"name": "FreshWave Body Wash Ocean 22 fl oz", "brand": "FreshWave", "department": "Beauty", "category": "Beauty", "subcategory": "Bath", "aisle": "HB02", "bay": "01", "stock": 18, "price": 5.48, "notes": "Body wash shelf under bar soap.", "synonyms": ["body wash", "shower gel", "bath wash"], "size": "22 fl oz", "unit": "bottle"},
    {"name": "FreshWave Deodorant Clean 2.6 oz", "brand": "FreshWave", "department": "Beauty", "category": "Beauty", "subcategory": "Deodorant", "aisle": "HB03", "bay": "04", "stock": 31, "price": 3.98, "notes": "Deodorant wall near travel sizes.", "synonyms": ["deodorant", "clean deodorant", "antiperspirant"], "size": "2.6 oz", "unit": "stick"},
    {"name": "SunGuard SPF 50 Sunscreen Lotion 8 oz", "brand": "SunGuard", "department": "Beauty", "category": "Beauty", "subcategory": "Sun Care", "aisle": "HB04", "bay": "02", "stock": 12, "price": 9.96, "notes": "Seasonal sun care shelf.", "synonyms": ["sunscreen", "spf 50", "sunblock"], "size": "8 oz", "unit": "bottle"},
    {"name": "CarePlus Flexible Fabric Bandages 100 Count", "brand": "CarePlus", "department": "Health", "category": "Health", "subcategory": "First Aid", "aisle": "H06", "bay": "02", "stock": 25, "price": 4.48, "notes": "First aid wall by gauze.", "synonyms": ["bandages", "band aids", "fabric bandages"], "size": "100 count", "unit": "box"},
    {"name": "CarePlus Digital Thermometer", "brand": "CarePlus", "department": "Health", "category": "Health", "subcategory": "Health Monitors", "aisle": "H06", "bay": "06", "stock": 7, "price": 8.98, "notes": "Thermometers in locked peg row.", "synonyms": ["thermometer", "digital thermometer", "fever thermometer"], "size": "standard", "unit": "each"},
    {"name": "MedEase Ibuprofen Tablets 200 Count", "brand": "MedEase", "department": "Pharmacy", "category": "Health", "subcategory": "Pain Relief", "aisle": "H08", "bay": "03", "stock": 14, "price": 7.98, "notes": "Pain relief wall beside acetaminophen.", "synonyms": ["ibuprofen", "pain medicine", "pain relief tablets"], "size": "200 count", "unit": "bottle"},
    {"name": "HoneyCough Drops Menthol 80 Count", "brand": "HoneyCough", "department": "Pharmacy", "category": "Health", "subcategory": "Cold and Flu", "aisle": "H09", "bay": "02", "stock": 18, "price": 3.76, "notes": "Cold and flu shelf near lozenges.", "synonyms": ["cough drops", "throat lozenges", "menthol drops"], "size": "80 count", "unit": "bag"},
    {"name": "Purrfect Feast Indoor Cat Food 16 lb", "brand": "Purrfect Feast", "department": "Pets", "category": "Pets", "subcategory": "Cat Food", "aisle": "PT02", "bay": "03", "stock": 12, "price": 21.98, "notes": "Cat food bags on lower shelf.", "synonyms": ["cat food", "indoor cat food", "pet food"], "size": "16 lb", "unit": "bag"},
    {"name": "CleanPaws Clumping Cat Litter 20 lb", "brand": "CleanPaws", "department": "Pets", "category": "Pets", "subcategory": "Cat Litter", "aisle": "PT01", "bay": "02", "stock": 16, "price": 11.98, "notes": "Heavy litter bags on bottom shelf.", "synonyms": ["cat litter", "clumping litter", "kitty litter"], "size": "20 lb", "unit": "bag"},
    {"name": "HappyTail Beef Dog Treats 24 oz", "brand": "HappyTail", "department": "Pets", "category": "Pets", "subcategory": "Dog Treats", "aisle": "PT03", "bay": "04", "stock": 21, "price": 8.98, "notes": "Treat pouches above large dog food bags.", "synonyms": ["dog treats", "pet treats", "training treats"], "size": "24 oz", "unit": "bag"},
    {"name": "PawChoice Adult Chicken Dog Food 35 lb", "brand": "PawChoice", "department": "Pets", "category": "Pets", "subcategory": "Dog Food", "aisle": "PT03", "bay": "02", "stock": 13, "price": 24.98, "notes": "Large adult dog food bags on the lower shelf.", "synonyms": ["dog food", "adult dog food", "chicken dog food"], "size": "35 lb", "unit": "bag", "attributes": ["dry food", "adult"]},
    {"name": "PawChoice Grain Free Salmon Dog Food 24 lb", "brand": "PawChoice", "department": "Pets", "category": "Pets", "subcategory": "Dog Food", "aisle": "PT03", "bay": "03", "stock": 6, "price": 31.98, "notes": "Grain-free dog food shelf beside sensitive stomach formulas.", "synonyms": ["dog food", "grain free dog food", "salmon dog food"], "size": "24 lb", "unit": "bag", "attributes": ["dry food", "grain free"]},
    {"name": "HappyTail Puppy Chicken Dog Food 15 lb", "brand": "HappyTail", "department": "Pets", "category": "Pets", "subcategory": "Dog Food", "aisle": "PT04", "bay": "01", "stock": 9, "price": 18.98, "notes": "Puppy formula bags near training pads.", "synonyms": ["puppy food", "dog food", "puppy dog food"], "size": "15 lb", "unit": "bag", "attributes": ["dry food", "puppy"]},
    {"name": "Everyday Choice Wet Dog Food Chicken 12 Count", "brand": PRIVATE_LABEL, "department": "Pets", "category": "Pets", "subcategory": "Dog Food", "aisle": "PT04", "bay": "03", "stock": 18, "price": 11.98, "notes": "Canned wet dog food trays on the middle shelf.", "synonyms": ["wet dog food", "canned dog food", "dog food cans"], "size": "12 count", "unit": "case", "attributes": ["wet food", "chicken"]},
    {"name": "TinyCare Baby Formula Gentle 30 oz", "brand": "TinyCare", "department": "Baby", "category": "Baby", "subcategory": "Formula", "aisle": "B10", "bay": "02", "stock": 8, "price": 32.98, "notes": "Formula shelf near baby food.", "synonyms": ["baby formula", "gentle formula", "infant formula"], "size": "30 oz", "unit": "can"},
    {"name": "TinyCare Baby Lotion 18 fl oz", "brand": "TinyCare", "department": "Baby", "category": "Baby", "subcategory": "Baby Toiletries", "aisle": "B11", "bay": "05", "stock": 17, "price": 5.98, "notes": "Baby bath shelf beside wash.", "synonyms": ["baby lotion", "lotion for baby", "infant lotion"], "size": "18 fl oz", "unit": "bottle"},
    {"name": "TinyCare Pacifiers 2 Pack", "brand": "TinyCare", "department": "Baby", "category": "Baby", "subcategory": "Feeding", "aisle": "B11", "bay": "01", "stock": 14, "price": 4.98, "notes": "Pacifier pegs near bottles.", "synonyms": ["pacifiers", "baby pacifier", "binky"], "size": "2 pack", "unit": "pack"},
    {"name": "Everyday Choice White Crew Socks 10 Pair", "brand": PRIVATE_LABEL, "department": "Apparel", "category": "Apparel", "subcategory": "Socks", "aisle": "C01", "bay": "02", "stock": 24, "price": 9.98, "notes": "Basic socks wall, white crew section.", "synonyms": ["socks", "white socks", "crew socks"], "size": "10 pair", "unit": "pack"},
    {"name": "Everyday Choice Cotton T-Shirt Black Large", "brand": PRIVATE_LABEL, "department": "Apparel", "category": "Apparel", "subcategory": "Basics", "aisle": "C02", "bay": "04", "stock": 18, "price": 7.98, "notes": "Folded basics table by size.", "synonyms": ["black t shirt", "cotton shirt", "large shirt"], "size": "large", "unit": "each"},
    {"name": "ComfortStep Memory Foam Slippers Medium", "brand": "ComfortStep", "department": "Apparel", "category": "Apparel", "subcategory": "Footwear", "aisle": "C04", "bay": "03", "stock": 9, "price": 14.98, "notes": "Slippers rack by sleepwear.", "synonyms": ["slippers", "house shoes", "memory foam slippers"], "size": "medium", "unit": "pair"},
    {"name": "AutoPure Windshield Washer Fluid 1 Gallon", "brand": "AutoPure", "department": "Automotive", "category": "Automotive", "subcategory": "Fluids", "aisle": "A20", "bay": "01", "stock": 26, "price": 3.48, "notes": "Automotive fluids shelf by funnels.", "synonyms": ["windshield washer fluid", "washer fluid", "wiper fluid"], "size": "1 gallon", "unit": "jug"},
    {"name": "RoadMax Full Synthetic Motor Oil 5W-30 5 qt", "brand": "RoadMax", "department": "Automotive", "category": "Automotive", "subcategory": "Motor Oil", "aisle": "A20", "bay": "04", "stock": 11, "price": 24.98, "notes": "Motor oil section arranged by viscosity.", "synonyms": ["motor oil", "5w30 oil", "synthetic oil"], "size": "5 quart", "unit": "jug"},
    {"name": "AutoGrip Tire Pressure Gauge", "brand": "AutoGrip", "department": "Automotive", "category": "Automotive", "subcategory": "Tools", "aisle": "A21", "bay": "03", "stock": 13, "price": 4.98, "notes": "Small automotive tools pegboard.", "synonyms": ["tire gauge", "pressure gauge", "tire pressure"], "size": "standard", "unit": "each"},
    {"name": "FitLife Yoga Mat 6 mm", "brand": "FitLife", "department": "Sports", "category": "Sports", "subcategory": "Fitness", "aisle": "S01", "bay": "02", "stock": 12, "price": 16.98, "notes": "Fitness aisle, rolled mats bin.", "synonyms": ["yoga mat", "exercise mat", "fitness mat"], "size": "6 mm", "unit": "each"},
    {"name": "PlayPro Soccer Ball Size 5", "brand": "PlayPro", "department": "Sports", "category": "Sports", "subcategory": "Team Sports", "aisle": "S02", "bay": "01", "stock": 14, "price": 12.98, "notes": "Ball rack near basketballs.", "synonyms": ["soccer ball", "football", "size 5 soccer ball"], "size": "size 5", "unit": "each"},
    {"name": "CampBright LED Camping Lantern", "brand": "CampBright", "department": "Outdoor", "category": "Outdoor", "subcategory": "Camping", "aisle": "S04", "bay": "05", "stock": 10, "price": 19.98, "notes": "Camping accessories shelf by flashlights.", "synonyms": ["camping lantern", "lantern", "led lantern"], "size": "standard", "unit": "each"},
    {"name": "GardenEase Nitrile Garden Gloves Medium", "brand": "GardenEase", "department": "Garden", "category": "Garden", "subcategory": "Gloves", "aisle": "GDN1", "bay": "02", "stock": 23, "price": 5.98, "notes": "Garden hand tools rack.", "synonyms": ["garden gloves", "gloves", "nitrile gloves"], "size": "medium", "unit": "pair"},
    {"name": "StickUp Removable Wall Hooks 6 Count", "brand": "StickUp", "department": "Home Improvement", "category": "Home", "subcategory": "Hanging Hardware", "aisle": "H16", "bay": "02", "stock": 18, "price": 6.98, "notes": "Adhesive hooks near picture hanging hardware.", "synonyms": ["wall hooks", "adhesive hooks", "removable hooks"], "size": "6 count", "unit": "pack"},
    {"name": "FixIt Duct Tape Silver 60 yd", "brand": "FixIt", "department": "Home Improvement", "category": "Home", "subcategory": "Tape", "aisle": "H16", "bay": "05", "stock": 20, "price": 5.48, "notes": "Tape and adhesive section.", "synonyms": ["duct tape", "silver tape", "repair tape"], "size": "60 yard", "unit": "roll"},
    {"name": "FreshFarm Roma Tomatoes 1 lb", "brand": "FreshFarm", "department": "Produce", "category": "Grocery", "subcategory": "Produce", "aisle": "P01", "bay": "Table 2", "stock": 38, "price": 1.78, "notes": "Loose tomatoes on produce table.", "synonyms": ["tomatoes", "roma tomatoes", "fresh tomatoes"], "size": "1 lb", "unit": "bag"},
    {"name": "FreshFarm Yellow Onions 3 lb Bag", "brand": "FreshFarm", "department": "Produce", "category": "Grocery", "subcategory": "Produce", "aisle": "P01", "bay": "Table 5", "stock": 22, "price": 2.78, "notes": "Bagged onions below potatoes.", "synonyms": ["onions", "yellow onions", "onion bag"], "size": "3 lb", "unit": "bag"},
    {"name": "FreshFarm Russet Potatoes 5 lb Bag", "brand": "FreshFarm", "department": "Produce", "category": "Grocery", "subcategory": "Produce", "aisle": "P01", "bay": "Table 6", "stock": 20, "price": 3.48, "notes": "Bagged potatoes near onions.", "synonyms": ["potatoes", "russet potatoes", "potato bag"], "size": "5 lb", "unit": "bag"},
    {"name": "GreenLeaf Baby Spinach 10 oz", "brand": "GreenLeaf", "department": "Produce", "category": "Grocery", "subcategory": "Packaged Produce", "aisle": "P02", "bay": "Cooler 4", "stock": 17, "price": 3.98, "notes": "Packaged greens cooler.", "synonyms": ["spinach", "baby spinach", "fresh spinach"], "size": "10 oz", "unit": "bag"},
    {"name": "MorningRise Orange Juice 52 fl oz", "brand": "MorningRise", "department": "Dairy", "category": "Grocery", "subcategory": "Juice", "aisle": "D02", "bay": "Cooler 8", "stock": 18, "price": 4.48, "notes": "Refrigerated juice shelf by milk.", "synonyms": ["orange juice", "juice", "refrigerated juice"], "size": "52 fl oz", "unit": "bottle"},
    {"name": "Creamery Greek Yogurt Vanilla 32 oz", "brand": "Creamery", "department": "Dairy", "category": "Grocery", "subcategory": "Yogurt", "aisle": "D04", "bay": "Cooler 6", "stock": 16, "price": 5.48, "notes": "Large yogurt tubs on lower shelf.", "synonyms": ["greek yogurt", "vanilla yogurt", "yogurt tub"], "size": "32 oz", "unit": "tub"},
    {"name": "BakeHouse Hamburger Buns 8 Count", "brand": "BakeHouse", "department": "Bakery", "category": "Grocery", "subcategory": "Bakery", "aisle": "B01", "bay": "07", "stock": 26, "price": 2.28, "notes": "Bun shelf beside sandwich bread.", "synonyms": ["hamburger buns", "buns", "burger buns"], "size": "8 count", "unit": "pack"},
    {"name": "BakeHouse Chocolate Chip Muffins 4 Count", "brand": "BakeHouse", "department": "Bakery", "category": "Grocery", "subcategory": "Bakery", "aisle": "B03", "bay": "Table 1", "stock": 14, "price": 4.98, "notes": "Bakery table near donuts.", "synonyms": ["muffins", "chocolate chip muffins", "bakery muffins"], "size": "4 count", "unit": "pack"},
    {"name": "QuickMeal Chicken Noodle Soup 18.6 oz", "brand": "QuickMeal", "department": "Pantry", "category": "Grocery", "subcategory": "Soup", "aisle": "G06", "bay": "02", "stock": 34, "price": 2.18, "notes": "Canned soup shelf.", "synonyms": ["soup", "chicken noodle soup", "canned soup"], "size": "18.6 oz", "unit": "can"},
    {"name": "QuickMeal Black Beans 15 oz", "brand": "QuickMeal", "department": "Pantry", "category": "Grocery", "subcategory": "Canned Goods", "aisle": "G06", "bay": "07", "stock": 40, "price": 1.12, "notes": "Bean section near canned vegetables.", "synonyms": ["black beans", "beans", "canned beans"], "size": "15 oz", "unit": "can"},
    {"name": "GoldenHarvest Jasmine Rice 5 lb", "brand": "GoldenHarvest", "department": "Pantry", "category": "Grocery", "subcategory": "Rice and Grains", "aisle": "G07", "bay": "01", "stock": 24, "price": 6.98, "notes": "Rice shelf beside pasta.", "synonyms": ["jasmine rice", "rice", "white rice"], "size": "5 lb", "unit": "bag"},
    {"name": "GoldenHarvest Extra Virgin Olive Oil 25.5 fl oz", "brand": "GoldenHarvest", "department": "Pantry", "category": "Grocery", "subcategory": "Cooking Oil", "aisle": "G08", "bay": "09", "stock": 15, "price": 8.98, "notes": "Cooking oil shelf below vinegar.", "synonyms": ["olive oil", "cooking oil", "extra virgin olive oil"], "size": "25.5 fl oz", "unit": "bottle"},
    {"name": "SpiceTrail Ground Cinnamon 2.37 oz", "brand": "SpiceTrail", "department": "Baking", "category": "Grocery", "subcategory": "Spices", "aisle": "G08", "bay": "08", "stock": 13, "price": 2.98, "notes": "Spice rack alphabetized near vanilla.", "synonyms": ["cinnamon", "ground cinnamon", "spice"], "size": "2.37 oz", "unit": "jar"},
    {"name": "MorningBowl Raisin Bran Cereal 18.7 oz", "brand": "MorningBowl", "department": "Breakfast", "category": "Grocery", "subcategory": "Cereal", "aisle": "G03", "bay": "05", "stock": 20, "price": 4.98, "notes": "Cereal aisle middle shelf.", "synonyms": ["raisin bran", "cereal", "bran cereal"], "size": "18.7 oz", "unit": "box"},
    {"name": "MorningBowl Instant Oatmeal Maple 10 Count", "brand": "MorningBowl", "department": "Breakfast", "category": "Grocery", "subcategory": "Oatmeal", "aisle": "G03", "bay": "08", "stock": 18, "price": 3.48, "notes": "Instant oatmeal packets beside oats.", "synonyms": ["instant oatmeal", "maple oatmeal", "oatmeal packets"], "size": "10 count", "unit": "box"},
    {"name": "SnackTime Pretzel Twists 16 oz", "brand": "SnackTime", "department": "Snacks", "category": "Grocery", "subcategory": "Salty Snacks", "aisle": "G11", "bay": "06", "stock": 25, "price": 3.28, "notes": "Pretzels shelf beside tortilla chips.", "synonyms": ["pretzels", "pretzel twists", "snacks"], "size": "16 oz", "unit": "bag"},
    {"name": "SnackTime Trail Mix 22 oz", "brand": "SnackTime", "department": "Snacks", "category": "Grocery", "subcategory": "Snack Mix", "aisle": "G12", "bay": "05", "stock": 18, "price": 6.98, "notes": "Trail mix pouch section.", "synonyms": ["trail mix", "snack mix", "nuts and raisins"], "size": "22 oz", "unit": "bag"},
    {"name": "ArcticBite Frozen Waffles 24 Count", "brand": "ArcticBite", "department": "Frozen", "category": "Grocery", "subcategory": "Frozen Breakfast", "aisle": "F03", "bay": "Freezer 2", "stock": 20, "price": 4.98, "notes": "Breakfast freezer beside pancakes.", "synonyms": ["frozen waffles", "waffles", "breakfast waffles"], "size": "24 count", "unit": "box"},
    {"name": "ArcticBite Vanilla Ice Cream 48 oz", "brand": "ArcticBite", "department": "Frozen", "category": "Grocery", "subcategory": "Ice Cream", "aisle": "F01", "bay": "Freezer 5", "stock": 17, "price": 4.48, "notes": "Family-size ice cream shelf.", "synonyms": ["vanilla ice cream", "ice cream tub", "ice cream"], "size": "48 oz", "unit": "carton"},
    {"name": "FreshCut Salmon Fillets 1 lb", "brand": "FreshCut", "department": "Seafood", "category": "Grocery", "subcategory": "Seafood", "aisle": "M02", "bay": "Cooler 1", "stock": 6, "price": 10.98, "notes": "Seafood cooler, front row.", "synonyms": ["salmon", "salmon fillets", "fresh salmon"], "size": "1 lb", "unit": "tray"},
    {"name": "FreshCut Shrimp Cooked 12 oz", "brand": "FreshCut", "department": "Seafood", "category": "Grocery", "subcategory": "Seafood", "aisle": "M02", "bay": "Cooler 3", "stock": 9, "price": 8.98, "notes": "Packaged seafood cooler.", "synonyms": ["shrimp", "cooked shrimp", "seafood"], "size": "12 oz", "unit": "bag"},
    {"name": "DeliCraft Rotisserie Chicken", "brand": "DeliCraft", "department": "Deli", "category": "Grocery", "subcategory": "Prepared Foods", "aisle": "D06", "bay": "Hot Case 1", "stock": 5, "price": 6.98, "notes": "Hot deli case near prepared sides.", "synonyms": ["rotisserie chicken", "hot chicken", "deli chicken"], "size": "whole", "unit": "each"},
    {"name": "DeliCraft Macaroni Salad 16 oz", "brand": "DeliCraft", "department": "Deli", "category": "Grocery", "subcategory": "Prepared Foods", "aisle": "D06", "bay": "Cooler 4", "stock": 12, "price": 3.98, "notes": "Prepared salad cooler.", "synonyms": ["macaroni salad", "deli salad", "prepared salad"], "size": "16 oz", "unit": "tub"},
    {"name": "HydratePlus Electrolyte Drink Lemon 6 Pack", "brand": "HydratePlus", "department": "Beverages", "category": "Grocery", "subcategory": "Sports Drinks", "aisle": "G10", "bay": "03", "stock": 18, "price": 5.98, "notes": "Sports drink shelf by flavored water.", "synonyms": ["electrolyte drink", "sports drink", "hydration drink"], "size": "6 pack", "unit": "pack"},
    {"name": "FizzPop Sparkling Water Lime 12 Pack", "brand": "FizzPop", "department": "Beverages", "category": "Grocery", "subcategory": "Sparkling Water", "aisle": "G10", "bay": "06", "stock": 22, "price": 4.98, "notes": "Sparkling water stack.", "synonyms": ["sparkling water", "lime sparkling water", "seltzer"], "size": "12 pack", "unit": "case"},
]


def _brand_for(name: str) -> str:
    if name.startswith("Everyday Choice"):
        return PRIVATE_LABEL
    for prefix, brand in BRAND_OVERRIDES.items():
        if name.startswith(prefix):
            return brand
    return name.split()[0]


def _size_from_name(name: str) -> str:
    matches = re.findall(r"\b\d+(?:\.\d+)?(?:/\d+)?\s*(?:fl oz|oz|lb|qt|gallon|gallons|count|pack|pair|sheets|ft|sq ft|yd|mm|inch|in)\b", name, flags=re.IGNORECASE)
    return matches[-1] if matches else "standard"


def _availability_for(stock: int, reorder_point: int) -> str:
    if stock <= 0:
        return "out_of_stock"
    if stock <= reorder_point:
        return "low_stock"
    return "in_stock"


def _enrich_operational_fields(item: dict[str, Any]) -> dict[str, Any]:
    stock = int(item.get("stock", 0))
    reorder_point = int(item.get("reorderPoint", 4))
    status = _availability_for(stock, reorder_point)
    search_facets = [
        item.get("brand"),
        item.get("department"),
        item.get("category"),
        item.get("subcategory"),
        item.get("size"),
        item.get("temperature"),
        *item.get("attributes", []),
    ]
    query_hints = [*item.get("synonyms", []), *search_facets]
    item["availabilityStatus"] = status
    item["inventoryConfidence"] = item.get("inventoryConfidence") or (0.82 if status == "out_of_stock" else 0.9 if status == "low_stock" else 0.96)
    item["lastUpdated"] = item.get("lastUpdated") or "2026-07-05T20:00:00-04:00"
    item["locationHint"] = item.get("locationHint") or f"Aisle {item['aisle']}, bay {item['bay']}. {item['notes']}"
    item["shelfTags"] = list(dict.fromkeys(str(value) for value in search_facets if value))
    item["queryHints"] = list(dict.fromkeys(str(value) for value in query_hints if value))
    return item


def _normalize_existing(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    sku = str(result["sku"])
    result["sku"] = sku if sku.startswith("SHOP-") else re.sub(r"^[A-Z]+-", "SHOP-", sku)
    result["name"] = str(result["name"])
    result["synonyms"] = [str(value) for value in result.get("synonyms", [])]
    brand = result.get("brand") or _brand_for(result["name"])
    category, subcategory = DEPARTMENT_META.get(result.get("department", ""), (result.get("department", "General"), result.get("department", "General")))
    result.update(
        {
            "brand": brand,
            "category": result.get("category", category),
            "subcategory": result.get("subcategory", subcategory),
            "size": result.get("size", _size_from_name(result["name"])),
            "unit": result.get("unit", "each"),
            "shelfLevel": result.get("shelfLevel", "standard"),
            "temperature": result.get("temperature", "ambient"),
            "reorderPoint": result.get("reorderPoint", max(4, min(18, int(result.get("stock", 0)) // 2))),
            "dailyVelocity": result.get("dailyVelocity", round(max(0.2, min(9.5, int(result.get("stock", 0)) / 9)), 2)),
            "restockEta": result.get("restockEta", "check store receiving notes"),
            "fulfillment": result.get("fulfillment", ["in_store", "pickup"]),
            "attributes": result.get("attributes", []),
            "allergens": result.get("allergens", []),
            "substitutes": result.get("substitutes", []),
            "ageRestricted": result.get("ageRestricted", False),
        }
    )
    if any(word in result["department"].lower() for word in ("dairy", "meat", "deli", "produce", "frozen", "seafood")):
        result["temperature"] = "frozen" if result["department"] == "Frozen" else "refrigerated"
    if int(result.get("stock", 0)) == 0:
        result["restockEta"] = result.get("restockEta", "next truck expected tonight")
    return _enrich_operational_fields(result)


def _new_item(index: int, item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    result["sku"] = f"SHOP-{index:04d}"
    result.setdefault("size", _size_from_name(result["name"]))
    result.setdefault("unit", "each")
    result.setdefault("shelfLevel", "standard")
    result.setdefault("temperature", "ambient")
    result.setdefault("reorderPoint", max(4, min(18, int(result.get("stock", 0)) // 2)))
    result.setdefault("dailyVelocity", round(max(0.2, min(9.5, int(result.get("stock", 0)) / 9)), 2))
    result.setdefault("restockEta", "standard nightly replenishment")
    result.setdefault("fulfillment", ["in_store", "pickup"])
    result.setdefault("attributes", [])
    result.setdefault("allergens", [])
    result.setdefault("substitutes", [])
    result.setdefault("ageRestricted", False)
    if result["department"] in {"Dairy", "Meat", "Deli", "Produce", "Frozen", "Seafood"}:
        result["temperature"] = "frozen" if result["department"] == "Frozen" else "refrigerated"
    return _enrich_operational_fields(result)


def main() -> int:
    base_items = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    normalized = [_normalize_existing(item) for item in base_items]
    existing_names = {item["name"].lower() for item in normalized}
    next_index = 1001 + len(normalized)
    for item in ADDITIONAL_PRODUCTS:
        if item["name"].lower() in existing_names:
            continue
        normalized.append(_new_item(next_index, item))
        existing_names.add(item["name"].lower())
        next_index += 1

    INVENTORY_PATH.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(normalized), "path": str(INVENTORY_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
