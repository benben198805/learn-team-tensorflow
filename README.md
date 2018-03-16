Tensorflow Environment Setup
============================

* Install docker-machine & virtualbox
	- brew install docker-machine
	- install virtualbox by click [here](http://download.virtualbox.org/virtualbox/5.1.8/VirtualBox-5.1.8-111374-OSX.dmg)

* Create virtual machine
	- docker-machine create -d virtualbox --virtualbox-memory 8196 tensorflow

* Look the ip of the virtual machine
	- docker-machine ip tensorflow

* Add this command into you environment file
	- eval $(docker-machine env tensorflow)

* Running the Docker container from the TW AI Club repository
	- `docker run -d -p 8888:8888 -p 6006:6006 -v {your-path}/:/notebooks --name tensorflow gcr.io/tensorflow/tensorflow:1.5.0-rc0-py3`
	- if you don't have tensorflow docker image, just `docker pull gcr.io/tensorflow/tensorflow:1.5.0-rc0-py3`

* Go to: http://localhost:8888, and then you can do everything you want

* See Local Url and Token: git logs tensorflow

* You also can use other ways to setup environment, just reminder please provide more memory as the calculation need.

* And there is a link about this course in [udacity](https://classroom.udacity.com/courses/ud730). You can get more information from it.


Step
======

* Run udacity/1_notmnist.ipynb and then it will download the data we need and stored in data folder
* the study process: lr -> dnn -> cnn -> udacity

8NXRYY4Y4Y-eyJsaWNlbnNlSWQiOiI4TlhSWVk0WTRZIiwibGljZW5zZWVOYW1lIjoiVGhvdWdodFdvcmtzLCBJbmMiLCJhc3NpZ25lZU5hbWUiOiJRaWxvbmcgWWFuIiwiYXNzaWduZWVFbWFpbCI6InFseWFuQHRob3VnaHR3b3Jrcy5jb20iLCJsaWNlbnNlUmVzdHJpY3Rpb24iOiIiLCJjaGVja0NvbmN1cnJlbnRVc2UiOnRydWUsInByb2R1Y3RzIjpbeyJjb2RlIjoiSUkiLCJmYWxsYmFja0RhdGUiOiIyMDE4LTAzLTA5IiwicGFpZFVwVG8iOiIyMDE5LTAzLTA4In1dLCJoYXNoIjoiODI2MDAxMy8yNzM1MjAwIiwiZ3JhY2VQZXJpb2REYXlzIjo3LCJhdXRvUHJvbG9uZ2F0ZWQiOmZhbHNlLCJpc0F1dG9Qcm9sb25nYXRlZCI6ZmFsc2V9-MNKWchTmTKt79pfGjLZ+vWbvo8BtDNG+Mw6HV6HnYzXFa8H5Rdq1bKfkydmLB/lwLcxzZ0Xy+W2atWSUUYgCjH2mDT9UyjjJ2uXjcOsGb0pzrPqpSw8NPrVgUVoSZij6h2ALKgUapcOriCLFPM9+croBbK7pP7SFsciLksecfTazEVVsTrVeXa4FXp9DoPUzYUjfxAb73viiUbqMGnn3UfhpU89zo4anD16c2a7LiWmhGHGP3R+wYmyxFJwQcLQmiF6WR3kJ+l42EailMwlqb++BNefvA96hIunPK1OTW2ukFDWQIrsNfbgCFHx5PWdtQxi3H6SC1SBHZLYhZ/Nsrw==-MIIEPjCCAiagAwIBAgIBBTANBgkqhkiG9w0BAQsFADAYMRYwFAYDVQQDDA1KZXRQcm9maWxlIENBMB4XDTE1MTEwMjA4MjE0OFoXDTE4MTEwMTA4MjE0OFowETEPMA0GA1UEAwwGcHJvZDN5MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxcQkq+zdxlR2mmRYBPzGbUNdMN6OaXiXzxIWtMEkrJMO/5oUfQJbLLuMSMK0QHFmaI37WShyxZcfRCidwXjot4zmNBKnlyHodDij/78TmVqFl8nOeD5+07B8VEaIu7c3E1N+e1doC6wht4I4+IEmtsPAdoaj5WCQVQbrI8KeT8M9VcBIWX7fD0fhexfg3ZRt0xqwMcXGNp3DdJHiO0rCdU+Itv7EmtnSVq9jBG1usMSFvMowR25mju2JcPFp1+I4ZI+FqgR8gyG8oiNDyNEoAbsR3lOpI7grUYSvkB/xVy/VoklPCK2h0f0GJxFjnye8NT1PAywoyl7RmiAVRE/EKwIDAQABo4GZMIGWMAkGA1UdEwQCMAAwHQYDVR0OBBYEFGEpG9oZGcfLMGNBkY7SgHiMGgTcMEgGA1UdIwRBMD+AFKOetkhnQhI2Qb1t4Lm0oFKLl/GzoRykGjAYMRYwFAYDVQQDDA1KZXRQcm9maWxlIENBggkA0myxg7KDeeEwEwYDVR0lBAwwCgYIKwYBBQUHAwEwCwYDVR0PBAQDAgWgMA0GCSqGSIb3DQEBCwUAA4ICAQC9WZuYgQedSuOc5TOUSrRigMw4/+wuC5EtZBfvdl4HT/8vzMW/oUlIP4YCvA0XKyBaCJ2iX+ZCDKoPfiYXiaSiH+HxAPV6J79vvouxKrWg2XV6ShFtPLP+0gPdGq3x9R3+kJbmAm8w+FOdlWqAfJrLvpzMGNeDU14YGXiZ9bVzmIQbwrBA+c/F4tlK/DV07dsNExihqFoibnqDiVNTGombaU2dDup2gwKdL81ua8EIcGNExHe82kjF4zwfadHk3bQVvbfdAwxcDy4xBjs3L4raPLU3yenSzr/OEur1+jfOxnQSmEcMXKXgrAQ9U55gwjcOFKrgOxEdek/Sk1VfOjvS+nuM4eyEruFMfaZHzoQiuw4IqgGc45ohFH0UUyjYcuFxxDSU9lMCv8qdHKm+wnPRb0l9l5vXsCBDuhAGYD6ss+Ga+aDY6f/qXZuUCEUOH3QUNbbCUlviSz6+GiRnt1kA9N2Qachl+2yBfaqUqr8h7Z2gsx5LcIf5kYNsqJ0GavXTVyWh7PYiKX4bs354ZQLUwwa/cG++2+wNWP+HtBhVxMRNTdVhSm38AknZlD+PTAsWGu9GyLmhti2EnVwGybSD2Dxmhxk3IPCkhKAK+pl0eWYGZWG3tJ9mZ7SowcXLWDFAk0lRJnKGFMTggrWjV8GYpw5bq23VmIqqDLgkNzuoog==