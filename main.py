#%%
import torch
from other import Person, Male, print_person, People


#%%

tom = Person(70, "female")
print(tom)
#%%
people = People([Person(30, "male"), Person(70, "female"), Person(55, "male")])
len(people)
#%%
# print(tom)
a = [1, 2, 3, 4]
print(a[1:3])
print(a.__getitem__(slice(1, 3)))
#%%
print_person(tom)
# print(f"age: {tom.age}, gender: {tom.gender}, is_grumpy: {tom.is_grumpy}")
# %%
print(tom)
# %%
tim = Male(70)
multiple_age = tim(5)
print(multiple_age)
# %%
if __name__ == "__main__":
    print(5)

#%%
import torch

# x -> L(x)
# x <- x-lr*D_x(L)

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([1.0, 2.0, 3.0])
print(a)
print(b)
#%%
a2 = a**2
b2 = b**2
print(a2)
print(b2)
#%%
a_loss = torch.sum(a2)
b_loss = torch.sum(b2)
print(a_loss)
print(b_loss)
#%%
print(a.grad)
a_loss.backward()
print(a.grad)
#%%
lr = 0.01
a = a - lr * a.grad
#%%
from torch.optim import SGD

optimizer = SGD([a], lr=0.01)
optimizer.step()
#%%
net = Net()
optimizer = SGD(net.parameters(), lr=0.01)
optimizer
#%%
print(a.grad)
optimizer.zero_grad()
print(a.grad)