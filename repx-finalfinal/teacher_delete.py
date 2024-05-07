import shutil
import os

def delete_assignment(code):
    if not os.path.exists(code):
        # print("Error: No such assignment exists")
        return 0

    shutil.rmtree(code)
    return 1


# # code=input("Enter assignment code:")
# delete_assignment(code)