from django.shortcuts import render

# Create your views here.
def base(request):
    return render(request,"base.html")

def about(request):
    return render(request,"about.html")

def assignment(request):
    return render(request,"assignment.html")

def attendance(request):
    return render(request,"attendance.html")

def fees(request):
    return render(request,"fees.html")

def login(request):
    return render(request,"login.html")

def notice(request):
    return render(request,"notice.html")

def profile(request):
    return render(request,"profile.html")

def register(request):
    return render(request,"register.html")

def result(request):
    return render(request,"result.html")

def student(request):
    return render(request,"student.html")

def timetable(request):
    return render(request,"timetable.html")

