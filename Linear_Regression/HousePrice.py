import matplotlib.pyplot as plt
import matplotlib.patches as patches
import  random

def linear_regression(data):
    #initial w and b
    w=50
    b=20
    learning_rate = 1e-6

    epoch=1
    parameter_sum=0.0 #for regularization

    while epoch<=1000:
        #w,b=compute_gradient(data,w,b,learning_rate,parameter_sum)
        index_dict={}
        randNum=random.randint(0,len(data))
        if randNum not in index_dict.keys():
            index_dict[randNum]=1
            w, b = stochastic_gradient(data, w, b, learning_rate, random_num=randNum)
            epoch+=1
    print(w,b)
    return w,b

def compute_gradient(data,current_w,current_b,learning_rate,parameter_sum):
    w=current_w
    b=current_b
    #gradient descent
    w_gradient = 0.0
    b_gradient = 0.0

    lambda_value=2

    for index in range(len(data[0])):
        x = data[0][index]
        y = data[1][index]

        #y=wx+b
        #regularization
        w_gradient += (-x) * (y - (w * x + b))+parameter_sum*lambda_value
        b_gradient += (-(y - (w * x + b)))+parameter_sum*lambda_value

    current_w = w - learning_rate * w_gradient
    current_b = b - learning_rate * b_gradient
    return current_w,current_b

def stochastic_gradient(data,current_w,current_b,learning_rate,random_num):
    w = current_w
    b = current_b
    # gradient descent
    w_gradient = 0.0
    b_gradient = 0.0

    x = data[0][random_num]
    y = data[1][random_num]

    # y=wx+b
    # stochastic gradient descent
    w_gradient = (-x) * (y - (w * x + b))
    b_gradient = (-(y - (w * x + b)))

    current_w = w - learning_rate * w_gradient
    current_b = b - learning_rate * b_gradient
    return current_w, current_b

def showData(data1,data2):
    plt.plot(data1[0],data1[1],'ro-')
    plt.plot(data2[0],data2[1],'b*-')
    plt.title('Price-Area')
    plt.xlabel('Square Feet')
    plt.ylabel('House Price')
    red_patch=patches.Patch(color='red',label='Original Data')
    blue_patch=patches.Patch(color='blue',label='Estimated Data')
    plt.legend(handles=[red_patch,blue_patch])
    plt.show()

def self_made():
    #63.4% error for best condition
    X = [150, 200, 250, 300, 350, 400, 600]
    Y = [6450, 7450, 8450, 9450, 11450, 15450, 18450]
    original_data = (X, Y)

    w, b = linear_regression(original_data)
    Y_Estimare = [x * w + b for x in X]
    estimate_data = (X, Y_Estimare)

    error = 0.0
    for index in range(len(Y)):
        error += (abs(Y[index] - Y_Estimare[index]) / Y[index])
    error *= 100
    print('Error percentage : %f' % error)

    showData(original_data, estimate_data)


def use_sklearn():
    #43.5% error
    from sklearn import linear_model

    X = [[150], [200], [250], [300], [350], [400], [600]]
    Y = [6450, 7450, 8450, 9450, 11450, 15450, 18450]
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    w = reg.coef_[0]
    b = reg.intercept_
    Y_Estimate = [w * x[0] + b for x in X]
    original = ([x[0] for x in X], Y)
    estimate = ([x[0] for x in X], Y_Estimate)

    error = 0.0
    for index in range(len(Y)):
        error += (abs(Y[index] - Y_Estimate[index]) / Y[index])
    error *= 100
    print('Error percentage : %f' % error)

    showData(original, estimate)


if __name__ == '__main__':
   #use_sklearn()
    self_made()


