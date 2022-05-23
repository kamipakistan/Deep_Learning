# y_pred_class = model.predict_classes(x_test)
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)
confusion_matrix(y_test_class, y_pred_class)
