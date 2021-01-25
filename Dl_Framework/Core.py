import numpy as np


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
     if(self.autograd):return Tensor(self.data.dot(x.data),autograd=True,creators=[self,x],creation_op="mm")
     return Tensor(self.data.dot(x.data))



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
           
##            if(not(isinstance(self.layers[i],Tanh)or isinstance(self.layers[i],Sigmoid))):
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
        pred = model.forward(data)
        return criterion.forward(pred, target).getdata()[0] 
    
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
        while((loss_list[-1])>=( .001)):
            for i in range(batch_no):
                pred = model.forward(data)
                loss = criterion.forward(pred, target)
                loss.backward(Tensor(np.ones_like(loss.data)))
                optim.step()
            loss_list.append(loss.getdata().sum(0))
            counter+=1
            if(counter==validation_counter):
                counter=0
                l=model.validate(validation_data,validation_target)
                if(l>loss_list[-1]):
                    print("overfitting occured")
                    return [loss_list,model]
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
    if(self.autograd):
     return Tensor(1 / (1 + np.exp(-self.data)), autograd=True,creators=[self],creation_op="sigmoid")
    return Tensor(1 / (1 + np.exp(-self.data)))
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
data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[0],[1],[1],[1]]), autograd=True)
model = Sequential([Linear(2,3),Tanh(), Linear(3,1),Tanh(),Linear(1,1),Sigmoid()])
print(model.get_parameters())
epoch_no=4

[list1,model]=model.train(target,data,epoch_no,.7,10,Tensor(np.array([1,0])),Tensor(np.array([1]))    )

print(list1)
print(model.get_parameters())

print(model.test(Tensor(np.array([1,1]))))
"""

model.set_parameters([np.array([[ 0.91265744,  0.31657938, -0.05536269],
       [ 0.07600885, -1.52912355, -2.73284409]]), np.array([-0.23677041,  0.476007  ,  1.0378013 ]),
                      np.array([[ 0.17098545],
       [-1.21845245],
       [-2.13460949]]), np.array([-0.40417909]),np. array([[4.18488716]]), np.array([0.00480434])]

)


print(list1[0])
print(list1[1].get_parameters())
model.set_parameters([np.array([[ 0.13293618,  0.06617196, -0.09045228],
     [-1.49060729, -2.38127353,  2.67146208]]), np.array([ 0.35460126,  0.63050873, -0.75129923]), np.array([[ 0.89519504]
 ,  [ 1.42247941],
 [-1.62664179]]), np.array([0.24830589]), np.array([[-4.17021191]]), np.array([-0.02951014])])
 
data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor(np.array([[1],[0],[0],[1]]), autograd=True)
model=Sequential([Linear(2,3),Tanh(),Linear(3,1),Sigmoid()])
list1=model.train(target,data,4,.1)
print("###########################")
print(list1[1].get_parameters())
"""
