from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=7)


#get the predictions for the test data
predicted_classes = model.predict_classes(X_val)
predicted_classes=list(lb.inverse_transform(predicted_classes))
y_true=[]
#get the indices to be plotted
for j in range(len(y_val)):
    y_true.append(lb.inverse_transform([list(y_val[j]).index(1)])[0])



from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))


a=np.array([y_true,predicted_classes])
correct,incorrect=[],[]
for i in range(a.shape[1]):
    if a[0][i]==a[1][i]:
        correct.append(i)
    else:
        incorrect.append(i)


for i ,enu in enumerate(correct):
    if i<9 :
        plt.subplot(3,3,i+1)
        img = cv2.resize(X_val[enu], (256, 256))
        plt.imshow(img ,cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[enu], y_true[enu]))
        plt.show()

for i ,enu in enumerate(incorrect):
    if i<25:
        plt.subplot(5,5,i+1)
        plt.imshow(X_val[enu], cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[enu], y_true[enu]))
        plt.show()


for i ,enu in enumerate(incorrect):
    if i<25 and y_true[enu]=='MIDDLE':
        plt.subplot(5,5,i+1)
        plt.imshow(X_val[enu], cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[enu], y_true[enu]))
        plt.show()
