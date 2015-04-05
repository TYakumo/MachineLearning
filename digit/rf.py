from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
        
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=4)
    rf.fit(train, target)
    rfResult = rf.predict(test)

    #for output result format
    finalResult = [['ImageId','Label']]
    i = 1
    for x in rfResult:
        finalResult.append([str(i),str(int(x))])
        i = i+1

    savetxt('Data/submission2.csv', finalResult, fmt='%s,%s')

if __name__=="__main__":
    main()


