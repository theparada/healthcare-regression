import pandas as pd

def process(data):
    data["Name"] = data["Name"].str.lower() # names are initially not consistent

    #label text data
    data = data.replace({
        "Gender": {
            "Male": 0,
            "Female": 1
        },
        "Blood Type": {
            'A+': 0,
            'A-': 1,
            'B+': 2,
            'B-': 3,
            'AB+': 4,
            'AB-': 5,
            'O+': 6,
            'O-': 7
        },
        "Medical Condition": {
            'Cancer': 0,
            'Obesity': 1,
            'Diabetes': 2,
            'Asthma': 3,
            'Hypertension': 4,
            'Arthritis': 5
        },
        "Insurance Provider": {
            'Blue Cross': 0,
            'Medicare': 1,
            'Aetna': 2,
            'UnitedHealthcare': 3,
            'Cigna': 4
        },
        "Admission Type":{
            'Urgent': 0,
            'Emergency': 1,
            'Elective': 2
        },
        "Medication": {
            'Paracetamol': 0,
            'Ibuprofen': 1,
            'Aspirin': 2,
            'Penicillin': 3,
            'Lipitor': 4
        },
        "Test Results": {
            'Normal': 0,
            'Inconclusive': 1,
            'Abnormal': 2
        }
    })

    # calculate numbers of days
    data["Date of Admission"] = pd.to_datetime(data["Date of Admission"])
    data["Discharge Date"] = pd.to_datetime(data["Discharge Date"])
    data["Stay Duration"] = data["Discharge Date"] - data["Date of Admission"]
    data["Stay Duration"] = pd.to_numeric(data["Stay Duration"].dt.days, downcast = "integer")

    # choose columns for the model
    dataModel = data[['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results', 'Stay Duration']]
    targetModel = data[['Billing Amount']]

    return dataModel, targetModel

def splitData(data, train=0.5, val=0.3):
    train_num = round(len(data)*train)
    val_num = round(len(data)*val)

    trainSet = data[:train_num]
    valSet = data[train_num:val_num]
    testSet = data[val_num:]

    return trainSet, valSet, testSet
