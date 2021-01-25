
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from statistics import mean 
import warnings
warnings.filterwarnings("error")





# foundation classes
###################################################
class SGD:
    def __init__(self, parameters, alpha=0.1):
      self.parameters = parameters
      self.alpha = alpha
    def zero(self):
      for p in self.parameters:
          p.grad.data *= 0
    def step(self, zero=True):
     for p in self.parameters:
       p.data -= p.grad.data * self.alpha
       if(zero):
        p.grad.data *= 0

class Tensor (object):
    def __init__(self,data,autograd=False,    creators=None,creation_op=None,id=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if(id is None):
          id = np.random.randint(0,100000)
          self.id = id
        if(creators is not None):
         for c in creators:
           if(self.id not in c.children):
            c.children[self.id] = 1
           else:
            c.children[self.id] += 1


    def all_children_grads_accounted_for(self):
         for id,cnt in self.children.items():
          if(cnt != 0): return False

         return True

    def backward(self,grad=None, grad_origin=None):
     if(grad is None):
         grad = Tensor(np.ones_like(self.data))
     if(self.autograd):
      if(grad_origin is not None):
        if(self.children[grad_origin.id] == 0):raise Exception("cannot backprop more than once")
        else: self.children[grad_origin.id] -= 1
      if(self.grad is None):
        self.grad = grad
      else:
        self.grad += grad
      if(self.creators is not None and(self.all_children_grads_accounted_for() or grad_origin is None)):

           if(self.creation_op == "add"):
              self.creators[0].backward(self.grad, self)
              self.creators[1].backward(self.grad, self)

           if(self.creation_op == "sub"):
                new = Tensor(self.grad.data)
                self.creators[0].backward(new, self)
                new = Tensor(self.grad.__neg__().data)
                self.creators[1].backward(new, self)

           if(self.creation_op == "mul"):
                new = self.grad * self.creators[1]
                self.creators[0].backward(new , self)
                new = self.grad * self.creators[0]
                self.creators[1].backward(new, self)

           if(self.creation_op == "mm"):

                act = self.creators[0]
                weights = self.creators[1]
                new = self.grad.mm(weights.transpose())
                act.backward(new)
                new = self.grad.transpose().mm(act).transpose()
                weights.backward(new)
           if(self.creation_op == "transpose"):
             self.creators[0].backward(self.grad.transpose())
							
           if("sum" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                ds = self.creators[0].data.shape[dim]
                self.creators[0].backward(self.grad.expand(dim,ds))
           if("expand" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim))
           if(self.creation_op == "neg"):
                self.creators[0].backward(self.grad.__neg__())     

           if(self.creation_op == "sigmoid"):
            ones = Tensor(np.ones_like(self.grad.data))
            self.creators[0].backward(self.grad * (self * (ones - self)))
           if(self.creation_op == "relu"):
            self.creators[0].backward(self.grad * (self ))
           if(self.creation_op == "tanh"):
            ones = Tensor(np.ones_like(self.grad.data))
            self.creators[0].backward(self.grad * (ones - (self * self)))


            
                
    def __neg__(self):
        if(self.autograd): return Tensor(self.data * -1,autograd=True, creators=[self],creation_op="neg")
        return Tensor(self.data * -1)
    def __add__(self, other):
       if(self.autograd and other.autograd):
         return Tensor(self.data + other.data, autograd=True,creators=[self,other], creation_op="add")
       return Tensor(self.data + other.data)
    def  getdata(self):
            return (self.data)
    def __repr__(self):
       return str(self.data.__repr__())
    def __str__(self):
       return str(self.data.__str__())

    def __sub__(self, other):
      if(self.autograd and other.autograd):
        return Tensor(self.data - other.data,autograd=True,creators=[self,other],creation_op="sub")
      return Tensor(self.data - other.data)
    def __mul__(self, other):
     if(self.autograd and other.autograd):
         try:
            return Tensor(self.data * other.data,autograd=True,creators=[self,other],creation_op="mul")
            return Tensor(self.data * other.data)
         except(RuntimeWarning):
               other.data/=9875
               if(self.autograd and other.autograd):
                  return Tensor(self.data * other.data,autograd=True,creators=[self,other],creation_op="mul")
               return Tensor(self.data * other.data)
         
          
        
             
    def sum(self, dim):
      if(self.autograd):
       return Tensor(self.data.sum(dim),autograd=True,creators=[self],creation_op="sum_"+str(dim))
      return Tensor(self.data.sum(dim))
    def expand(self, dim,copies):
      trans_cmd = list(range(0,len(self.data.shape)))
      trans_cmd.insert(dim,len(self.data.shape))
      new_shape = list(self.data.shape) + [copies]
      new_data = self.data.repeat(copies).reshape(new_shape)
      new_data = new_data.transpose(trans_cmd)
      if(self.autograd):
         return Tensor(new_data,autograd=True,creators=[self],creation_op="expand_"+str(dim))
      return Tensor(new_data)

    def transpose(self):
     if(self.autograd):return Tensor(self.data.transpose(),autograd=True,creators=[self],creation_op="transpose")
     return Tensor(self.data.transpose())

    def mm(self, x):
     if(self.autograd):
        try:
         return Tensor(self.data.dot(x.data),autograd=True,creators=[self,x],creation_op="mm")
        except(TypeError):
            print(x.data)
            print(x.data.shape)
            print(self.data.shape)
            
     return Tensor(self.data.dot(x.data))
    
        
    def shape(self):
        return self.data.shape


class Layer(object):
    def __init__(self):
     self.parameters = list()
    def get_parameters(self):
      return self.parameters
    def change_Weights(self,para):
          print( self.parameters )
          print(type(self.parameters))


class Linear(Layer):
      def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs)*np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)
      def forward(self, input):
         return input.mm(self.weight)+self.bias.expand(0,len(input.data))
      def change_Weights(self,para):
            self.weight = Tensor(para, autograd=True)
            self.parameters=[]
            self.parameters.append(self.weight)
            self.parameters.append(self.bias)

            
class MSELoss(Layer):
            def __init__(self):
               super().__init__()
            def forward(self, pred, target):
              return ((pred - target)*(pred - target)).sum(0)


class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers
    def add(self, layer):
        self.layers.append(layer)
    def forward(self, input):
      for layer in self.layers:
        input = layer.forward(input)
      return input
    def set_parameters(self,para):
       # print(isinstance(self.layers[0],Linear))#is instance  to compare class tpes

        for i in range (len(self.layers)):
           
            #if(not(isinstance(self.layers[i],Tanh)or isinstance(self.layers[i],Sigmoid))):
             self.layers[i].change_Weights(para[i])

    def get_parameters(self):
         params = list()
         for l in self.layers:
          params+= l.get_parameters()
         
         return params

    def validate(self,data,target):
        criterion = MSELoss()
        optim = SGD(parameters=model.get_parameters(),alpha=1)
        loss_list=[]
        for i in data:
         pred = model.forward(i)
         loss_list.append(criterion.forward(pred, target).getdata()[0])

        return mean(loss_list) 
    
    def train(model,target,data,batch_no,alpha,validation_counter,validation_data,validation_target):
        criterion = MSELoss()
        optim = SGD(parameters=model.get_parameters(),alpha=alpha)
        loss_list=[]
        pred = model.forward(data)
        loss = criterion.forward(pred, target)
        loss.backward(Tensor(np.ones_like(loss.data)))
        optim.step()
        loss_list.append(loss.getdata()[0] )
        validation_list=[]
        counter=0
        count=0
        while((loss_list[-1])>=( .001) and count<100 ):
            for i in range(batch_no):
                pred = model.forward(data)
                loss = criterion.forward(pred, target)
                loss.backward(Tensor(np.ones_like(loss.data)))
                optim.step()
            loss_list.append(loss.getdata().sum(0))
            count+=1
            print(count)
            if(len(loss_list)>20):
                if(mean(loss_list[-10:-2])<=loss_list[-1] or (loss_list[-1]==loss_list[-2] and loss_list[-2]==loss_list[-3])):
                    print("training reached overflow")
                    return [loss_list,model]
##            counter+=1
              
##            if(counter==validation_counter):
##                counter=0
##                l=model.validate(validation_data,validation_target)
##                if(l>loss_list[-1]):
##                    print("overfitting occured")
##                    return [loss_list,model]
           
        return [loss_list,model]



    def test(self,data):

        return   self.forward(data)
        
 
####################################################3
#activation functions syntax pass numpy array return a numpy array
#############################################
def hard_sigmoid(x):
    l=np.zeros(len(x))
    for i in  range (len(x)):
	    if(x[i]>1): l[i]=1
	    elif(x[i]<=0): l[i]=0
	    else:l[i]=(x[i]+1)/2
    return l



def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_




def leaky_relu_function(x):
    if x<0:
        return 0.01*x
    else:
        return x
def parametrized_relu_function(a,x):
    if x<0:
        return a*x
    else:
        return x
def elu_function(x, a):
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x
    
def swish_function(x):
    return x/(1-np.exp(-x))


 #############################################
# activation functions that call tensor
 ####################################### #####      
def sigmoid(self):
              try:             
                if(self.autograd):
                   return Tensor(1 / (1 + np.exp(-self.data)), autograd=True,creators=[self],creation_op="sigmoid")
                return Tensor(1 / (1 + np.exp(-self.data)))
              except(RuntimeWarning):       
                 if(self.autograd):
                     return Tensor(np.array(.5), autograd=True,creators=[self],creation_op="sigmoid")
                 return Tensor(np.array(.5))
                  

                
     
                          
             



def tanh(self):
 if(self.autograd):
   return Tensor(np.tanh(self.data),autograd=True,creators=[self],creation_op="tanh")
 return Tensor(np.tanh(self.data))        

def relu1(self):
 if(self.autograd):
   return Tensor(   np.maximum(0, self.data),autograd=True,creators=[self],creation_op="relu")
 return Tensor(np.maximum(0, self.data))        

def leaky_relu(self):
 if(self.autograd):
   return Tensor(np.maximum(self.data *.001, self.data),autograd=True,creators=[self],creation_op="relu")
 return Tensor(np.maximum( self.data*.01, self.data))        

########################################################
#activation functions classes
########################################################
class Sigmoid(Layer):
   def __init__(self):
    super().__init__()
   def forward(self, input):
    return sigmoid(input)
class Tanh(Layer):
  def __init__(self):
   super().__init__()
  def forward(self, input):
   return tanh(input)

class Relu(Layer):
  def __init__(self):
   super().__init__()
  def forward(self, input):
   return relu1(input)


class LeakyRelu(Layer):
  def __init__(self):
   super().__init__()
  def forward(self, input):
   return leaky_relu(input)


##################################################
#main part of code
################################################

#load teat data and train data
train_file = pd.read_csv('train.csv')
train_file = train_file.sample(frac=1)
train_set = train_file[0:600]
test_set = train_file[600:700]
validat_set=train_file[500:600]
#creat Y_train which have only the label Column of the train.csv file
Y_train = train_set["label"]
Y_test = test_set["label"]
Y_valid=validat_set["label"]
# creat X_train which have all columns of the train.csv file Except 'label' column
#X_train = train_file.drop(labels = ["label"],axis = 1)
X_train = train_set.drop(["label"],axis = 1)
X_test = test_set.drop(["label"],axis = 1)
X_valid=validat_set.drop(["label"],axis = 1)
#print(X_train)
print('-----------------------------')
# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0
X_valid=X_valid/255.0


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
#X_train = X_train.values.reshape(-1,28,28,1)
#X_test = X_test.values.reshape(-1,28,28,1)
#X_valid = X_valid.values.reshape(-1,28,28,1)

print('--------------- --------------')
arr2=np.array(Y_train.values)
print(arr2.shape)

##arr1=np.array(X_train)
##arr5=np.array(X_valid)
##print(type(arr1))
##print(arr1[0].shape)
##print(np.array([[0,0],[0,1],[1,0],[1,1]]).shape)
##print(np.array([[0,0],[0,1],[1,0],[1,1]])[0].shape)


def transformer(x):
    arr=[]
    for i in x:
        arr.append(np.array([i]))
    return arr    
        
arr1=np.array((X_train.values))
arr2=np.array(transformer(Y_train.values))
arr3=np.array((X_test.values))
arr4=np.array(transformer(Y_test.values))

arr5=np.array((X_valid.values))    
arr6=np.array(transformer(Y_valid.values))

print((arr1[0]))
print(arr2)



##print(np.array(Y_valid.values).shape)
##print(arr6[0].shape)
##print(arr6[1].shape)
##print(np.array(Y_valid.values)[2].shape)
##
##print(np.array([[0],[1],[1],[1]]).shape)
##print(np.array([[0],[1],[1],[1]])[0].shape)
##print(np.array([[0],[1],[1],[1]])[1].shape)
##print(np.array([[0],[1],[1],[1]])[2].shape)
##
data = Tensor(arr1, autograd=True)
target = Tensor(arr2, autograd=True)
model = Sequential([Linear(784,400),Tanh(),Linear(400,90),Tanh(),Linear(90,1),Tanh()])
epoch_no=20
[list1,model]=model.train(target,data,epoch_no,1,50,Tensor(arr5),Tensor(arr6))    

print(list1)
print("***********************************")
print(model.test(Tensor(arr3[0])))
print(arr4[0])



print(model.test(Tensor(arr3[1])).sum(0))
print(arr4[1])


print(model.test(Tensor(arr3[2])))
print(arr4[2])
print(model.test(Tensor(arr3[3])))
print(arr4[3])
