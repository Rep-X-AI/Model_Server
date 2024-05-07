import shutil
import os

def delete_assignment(code):
    if not os.path.exists(code):
        print("Error: No such assignment exists")
        return

    shutil.rmtree(code)


code=input("Enter assignment code:")
delete_assignment(code)