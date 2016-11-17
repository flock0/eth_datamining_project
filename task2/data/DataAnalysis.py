import numpy as np
import sys
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if len(sys.argv) <= 1:
	print("Not enough argument")

# Load the test file
input_matrix = np.genfromtxt('handout_test.txt', delimiter=' ')

# Extract the y and x
x = input_matrix[:,1:]
y = input_matrix[:,0].reshape(20000,1)
y_raw = input_matrix[:,0]

# Do the transformation
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, interaction_only=True)
#selection = SelectKBest(k=50)
n_components = int(sys.argv[1])
selection = PCA(n_components=n_components)

x_scale = scaler.fit_transform(x,y_raw)
x_trans = selection.fit_transform(x_scale,y_raw)
x_poly = poly.fit_transform(x_trans,y_raw)
x_transform = x_poly

# Do the regression with a linear regressor<html>
<head>
<title>Untitled</title>
</head>
Dear jane <br>
<p>You are invited at the weekly meeting
<p>Yours sincerely, <br>
John
</html>

#regr = linear_model.LinearRegression()
regr = SVC(kernel='linear')
predicted = cross_val_predict(regr, x_transform, y_raw, cv=10)


# Evaluate the score

FP = 0
FN = 0
TP = 0
TN = 0
prediction = np.sign(predicted)
for i in range(x_trans.shape[0]):
    if prediction[i] == y[i] and y[i] == 1:
        TP = TP + 1
    elif prediction[i] == y[i] and y[i] == -1:
        TN = TN + 1
    elif prediction[i] != y[i] and y[i] == 1:
        FN = FN + 1
    elif prediction[i] != y[i] and y[i] == -1:
        FP = FP + 1
    else:
        print("Something went wrong:", prediction[i], y)
    
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("accuracy:", (TP + TN)/x_trans.shape[0])
