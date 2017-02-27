forest=ff
forest.fit(X,y)
#feature_names = vectorizer.get_feature_names()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

output_f=open('important features_newdata_new features_all.txt','wb')
output_f.write(str(forest))
output_f.write('\n')
output_f.write('\n')

for f in range(X.shape[1]):
    print("%d. feature %s (%f) " % (f + 1,feature_names[indices[f]] , importances[indices[f]]))
    output_f.write("%d. feature %s (%f) " % (f + 1,feature_names[indices[f]] , importances[indices[f]]))
    output_f.write('\n')
    #print("%d. feature %s (%f)  %d  %d" % (f + 1,feature_names[indices[f]] , importances[indices[f]],  M_dic[feature_names[indices[f]]], B_dic[feature_names[indices[f]]]))

output_f.close()
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
#number=X.shape[1]
number=20
plt.bar(range(number), importances[indices[0:number]],
       color="r", yerr=std[indices[0:number]], align="center")
plt.xticks(range(number), indices)
plt.xlim([-1, number])
plt.show()
