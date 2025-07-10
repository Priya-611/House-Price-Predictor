import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  #it works best with mixed data(both categorical and numerical features) [handles non-linear patterns as car price don't increase in straight line]
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

df=pd.read_csv(r"C:\\Users\HP\\OneDrive\\Documents\\Project(Data Science)\\House\\Housing.csv")
# print(df.head())

# print(df.info())

# print(df.describe())

# print(df.isnull().sum())


# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()


road_encoder=LabelEncoder()
guestroom_encoder=LabelEncoder()
basement_encoder=LabelEncoder()
hotwater_encoder=LabelEncoder()
ac_encoder=LabelEncoder()
prefarea_encoder=LabelEncoder()
furnishing_encoder=LabelEncoder()


df['mainroad']=road_encoder.fit_transform(df['mainroad'])
df['guestroom']=guestroom_encoder.fit_transform(df['guestroom'])
df['basement']=basement_encoder.fit_transform(df['basement'])
df['hotwaterheating']=hotwater_encoder.fit_transform(df['hotwaterheating'])
df['airconditioning']=ac_encoder.fit_transform(df['airconditioning'])
df['prefarea']=prefarea_encoder.fit_transform(df['prefarea'])
df['furnishingstatus']=furnishing_encoder.fit_transform(df['furnishingstatus'])

# sns.heatmap(df.corr(),annot=True)
# plt.show()


# print(df.columns)

# 'price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
#        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
#        'parking', 'prefarea', 'furnishingstatus'

x=df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'airconditioning', 'parking']]
y=df['price']



x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=45)
model=RandomForestRegressor()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mse=mean_squared_error(y_test,y_pred)   #average difference between predicted and actual ratings (lower is better)
r_sq=r2_score(y_test,y_pred)   #Closer to 1 is better[perfect prediction]

print("Mean Square Error: ",mse )
print("r2_square", r_sq)    


# print("Enter details: ")


with open('house_price_model.pkl','wb') as f:
    pickle.dump(model,f)

    
with open('main_road.pkl','wb') as f:
    pickle.dump(road_encoder,f)

with open('guest_room.pkl','wb') as f:
    pickle.dump(guestroom_encoder,f)

with open('basement.pkl','wb') as f:
    pickle.dump(basement_encoder,f)

with open('air_conditioning.pkl','wb') as f:
    pickle.dump(ac_encoder,f)

