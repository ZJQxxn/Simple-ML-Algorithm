'''
    The gradient descent made by myself have 
    78% error percentage.But by using sklearn
    the error decrease to about 35%.
'''
import matplotlib.pyplot as plt
from sklearn import linear_model

def linear_regression(data):
    #initial w and b
    w=2005
    b=0
    learning_rate = 5e-10

    epoch=1
    w_gradient=1
    b_gradient=1
    while epoch<=100000 or (w_gradient<0.01 and b_gradient<0.01):
        w,b,w_gradient,b_gradient=compute_gradient(data,w,b,learning_rate)
        epoch+=1
    print(w,b,epoch)
    return w,b

def compute_gradient(data,current_w,current_b,learning_rate):
    w=current_w
    b=current_b
    #gradient descent
    w_gradient = 0.0
    b_gradient = 0.0

    N=len(data[0])
    for index in range(N):
        x = data[0][index]
        y = data[1][index]
        #y=wx+b
        w_gradient += (1/N)*(-x) * (y - (w * x + b))
        b_gradient += (1/N)*(-(y - (w * x + b)))
    current_w = w - learning_rate * w_gradient
    current_b = b - learning_rate * b_gradient
    return current_w,current_b,w_gradient,b_gradient

def showData(data1,data2):
    plt.plot(data1[0],data1[1],'ro-',label='Original Data')
    plt.plot(data2[0],data2[1],'b*-',label='Estimate Data')
    plt.title('Physician-Year')
    plt.xlabel('Year')
    plt.ylabel('American Physician')
    #red_patch=patches.Patch(color='red',label='')
    #blue_patch=patches.Patch(color='blue',)
    #plt.legend(handles=[red_patch,blue_patch])
    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.show()

def readData():
    with open('AmericaPhysicianData.txt','r') as file:
        lines=file.readlines()

    X_Parameter=[]
    Y_Parameter=[]

    for each in lines:
        line_split=each.replace("\"","").split(';')
        X_Parameter.append(int(line_split[1]))
        Y_Parameter.append(int(line_split[2]))
    return X_Parameter,Y_Parameter

def self_made():
    X_Parameter, Y_Parameter = readData()
    originalData = (X_Parameter, Y_Parameter)

    w, b = linear_regression(originalData)
    Y_Estimate = [w * x + b for x in X_Parameter]
    estimateData = (X_Parameter, Y_Estimate)

    error = 0.0
    for index in range(len(Y_Parameter)):
        error += (abs(Y_Parameter[index] - Y_Estimate[index]) / Y_Parameter[index])
    error *= 100
    print('Error percentage : %f' % error)

    showData(originalData, estimateData)

def sklearn_made():
    X_Parameter, Y_Parameter = readData()
    originalData = (X_Parameter, Y_Parameter)
    X_Parameter=[[x] for x in X_Parameter]

    lr=linear_model.LinearRegression()
    lr.fit(X_Parameter,Y_Parameter)

    w=lr.coef_[0]
    b=lr.intercept_
    print(w,b)

    Y_Estimate = [w * x[0] + b for x in X_Parameter]
    estimateData = (X_Parameter, Y_Estimate)

    error = 0.0
    for index in range(len(Y_Parameter)):
        error += (abs(Y_Parameter[index] - Y_Estimate[index]) / Y_Parameter[index])
    error *= 100
    print('Error percentage : %f' % error)

    showData(originalData,estimateData)



if __name__ == '__main__':
    self_made()
    #sklearn_made()


