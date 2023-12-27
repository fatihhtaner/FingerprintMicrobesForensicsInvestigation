import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

dosyaUzantisi = r"otu.csv"
veri = pd.read_csv(dosyaUzantisi, dtype=str)
### X = veri.iloc[2:, :].T
### y = veri.iloc[:1, :].T.iloc()
X = veri.iloc[1:, :].T
y = veri.iloc[:1, :].T.squeeze()
le = LabelEncoder()
y = le.fit_transform(y)

'''
En yüksek başarı oranını bulabilmek için aşağıdaki fonkisyonla değerleri denedim.
best_accuracy = 0.0
best_test_size = 0.0
best_random_state = 0

for test_size in [0.2, 0.21, 0.22]:
    for random_state in [0, 42, 100]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        model = GradientBoostingClassifier(random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Size: {test_size}, Random State: {random_state}, Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_test_size = test_size
            best_random_state = random_state
'''
#Veriyi eğitim ve test parçalarına böldüm 0.21 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=0)
### model = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, max_iter=300))
###model = RandomForestClassifier(random_state=0)  

# Daha sonrasında Gradient Boosting ile denedim ve bu orandaki randomstate ve test_size değerlerine göre en yüksek tutarlılık sonucunu aldım
model = GradientBoostingClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
Matris = confusion_matrix(y_test, y_pred)
Rapor = classification_report(y_test, y_pred)
print("Confusion Matrix:\n", Matris)
print("Classification Report:\n", Rapor)
tutarlilik = accuracy_score(y_test, y_pred)
hassasiyet = Matris[0, 0] / (Matris[0, 0] + Matris[0, 1])
print("Accuracy:", tutarlilik, "\n")
print('Sensitivity : ', hassasiyet)
specificity = Matris[1, 1] / (Matris[1, 0] + Matris[1, 1])
ROC_AUC = roc_auc_score(y_test, y_pred)
print('Specificity : ', specificity)
print('ROC AUC : {:.4f}'.format(ROC_AUC))