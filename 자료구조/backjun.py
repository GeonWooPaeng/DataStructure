# 1.
# case_test = int(input())
# for _ in range(case_test):
#     L = list(input())
#     stack_l = []
#     stack_r = []

#     for i in range(len(L)):
#         stack_l.append(L[i])
#         if stack_l[-1] == '<':
#             stack_l.pop()
#             if stack_l:
#                 stack_r.append(stack_l.pop())

#         elif stack_l[-1] == '>':
#             stack_l.pop()
#             if stack_r:
#                 stack_l.append(stack_r.pop())
        
#         elif stack_l[-1] == '-':
#             stack_l.pop()
#             if stack_l:
#                 stack_l.pop()

#     stack_l.extend(reversed(stack_r))
#     print(''.join(stack_l))


# 2.
# test_case = int(input())

# for _ in range(test_case):
#     stack_l = []
#     stack_r = []
#     data = input() 
#     for i in data:
#         if i == '-':
#             if stack_l:
#                 stack_l.pop()
        
#         elif i == '<':
#             if stack_l:
#                 stack_r.append(stack_l.pop())
                
#         elif i == '>':
#             if stack_r:
#                 stack_l.append(stack_r.pop())
        
#         else:
#             stack_l.append(i)

#     stack_l.extend(reversed(stack_r))
#     print(''.join(stack_l))


# 1.
# import hashlib

# data = str(input())
# encoded_data = data.encode()
# result = hashlib.sha256(encoded_data).hexdigest()
# print(result)
