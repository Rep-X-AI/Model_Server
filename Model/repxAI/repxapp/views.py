from django.shortcuts import render
from .ml import get_answer

def input_form(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'POST':
        model_ans = request.POST['input1']
        scan_ans = request.POST['input2']
        diagram_url = request.POST['input3']
        result = get_answer(model_ans, scan_ans, diagram_url)
        return render(request, 'result.html', {'result': result})
    else:
        return render(request, 'home.html')