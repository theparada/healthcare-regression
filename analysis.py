from data_processing import process
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./healthcare_dataset.csv")
dataModel, targetModel = process(data)

print(dataModel)
print(targetModel)

# plt.hist2d(dataModel['Age'], labelModel["Billing Amount"], bins=50)
# plt.savefig('./images/age_v_bill.png')
