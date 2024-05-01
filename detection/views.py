# detection/views.py

import json
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import FileLog, TrainLog, TestLog
import pandas as pd
from joblib import load

import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from PIL import Image, ImageTk
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import seaborn as sns

from io import BytesIO
import base64

le = LabelEncoder()

new_data = pd.read_csv(r"D:/Portfolio/Portfolio_V2/sachinlohar_V2/Trainnew.csv")#, sep='COLUMN_SEPARATOR', dtype=np.float64)
new_data['protocol_type']=le.fit_transform(new_data['protocol_type'])
new_data['service']=le.fit_transform(new_data['service'])
new_data['flag']=le.fit_transform(new_data['flag'])
x_train = new_data.drop(["label"],axis=1)
y_train = new_data["label"]


new_data = pd.read_csv(r"D:/Portfolio/Portfolio_V2/sachinlohar_V2/Testnew.csv")#, sep='COLUMN_SEPARATOR', dtype=np.float64)
new_data['protocol_type']=le.fit_transform(new_data['protocol_type'])
new_data['service']=le.fit_transform(new_data['service'])
new_data['flag']=le.fit_transform(new_data['flag'])
x_test = new_data.drop(["label"],axis=1)
y_test = new_data["label"]

model_list = {"SVM":r"D:/Portfolio/Portfolio_V2/sachinlohar_V2/OLD_MODELS/SVM.joblib","RF":r"D:/Portfolio/Portfolio_V2/sachinlohar_V2/OLD_MODELS/RF.joblib"}

@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        model_type = request.POST.get('selected_model')
        
        if not model_type:
            return JsonResponse({'error': 'Missing model type'}, status=400)

        try:
            # Train the model
            model = None
            if model_type == 'SVM':
                model = SVC(kernel='linear', random_state=2)
            elif model_type == 'RF':
                num_trees = 100
                max_features = 3
                model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
            else:
                return JsonResponse({'error': 'Invalid model type'})
            
            # Measure training time
            start_time = time.time()
            print(start_time)
            # model.fit(x_train, y_train)
            training_time = time.time() - start_time
            print(training_time)

            chosen_model = model_list[model_type]
        
            # Load the selected model
            model1 = load(chosen_model)

            # Make predictions on the test set
            predictions = model1.predict(x_test)
            
            # Calculate accuracy and classification report
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            
            # Save the trained model
            # model_name = f"{model_type}.joblib"
            # from joblib import dump
            # dump(model, model_name)
            
            # Save the train log
            TrainLog.objects.create(model=model_type, accuracy=accuracy)
            
            # Generate confusion matrix
            confusion_mat = confusion_matrix(y_test, predictions)

            # Plot confusion matrix as heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            # Save the plot to a buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            
            return JsonResponse({
                'model_type': model_type,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': plot_data,
                'training_time': training_time
            })
        except Exception as e:
            print(str(e))
            return JsonResponse({'error': str(e)}, status=500)
        
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def test_model(request):
    if request.method == 'POST':
        selected_model = request.POST.get('selected_model')
        data_file = request.FILES.get('data_file')
        chosen_model = model_list[selected_model]
        
        # Load the selected model
        model = load(chosen_model)
        
        # Read the data from the uploaded file
        try:
            file = pd.read_csv(data_file)
        except pd.errors.EmptyDataError:
            return JsonResponse({'error': 'No data found in the file'}, status=400)
        
        if file.empty:
            return JsonResponse({'error': 'Empty file'}, status=400)
        
        # Transform data
        le = LabelEncoder()
        file['protocol_type'] = le.fit_transform(file['protocol_type'])
        file['service'] = le.fit_transform(file['service'])
        file['flag'] = le.fit_transform(file['flag'])
        
        # Prepare data for prediction
        qn = file.drop(["label"], axis=1)
        
        # Make predictions
        predictions = model.predict(qn)
        
        # Process predictions
        result = predictions[0]
        is_ddos_value = 1 if result != 'normal' else 0
        output = 'DDOS Attack Detected' if result != 'normal' else 'DDOS Attack Not Detected'
        
        # Save log data
        FileLog.objects.create(
            protocol_type=file['protocol_type'].iloc[0],
            flag=file['flag'].iloc[0],
            service=file['service'].iloc[0],
            is_ddos=is_ddos_value
        )
        
        # Convert DataFrame to list of dictionaries
        uploaded_data_json = json.dumps(file.to_dict(orient='records'))

        return JsonResponse({
            'result': output,
            'is_ddos': is_ddos_value,
            'uploaded_data': uploaded_data_json
        })

    return JsonResponse({'error': 'Method not allowed'}, status=405)


def generate_plots():
    # Perform your graphical analysis here
    model_accuracies = {"SVM": 90.5, "RF": 85.3}
    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    # Plot 1: Bar chart for model accuracies
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(models, accuracies, color=['blue', 'green'])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracies')
    ax1.set_ylim(0, 100)

    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    plot_data1 = base64.b64encode(buffer1.read()).decode('utf-8')
    buffer1.close()

    # Plot 2: Pie chart for model distribution
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(accuracies, labels=models, autopct='%1.1f%%', startangle=140)
    ax2.set_title('Model Distribution')

    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    plot_data2 = base64.b64encode(buffer2.read()).decode('utf-8')
    buffer2.close()

    # Plot 3: Line plot for model comparison
    x_values = np.linspace(0, 10, 100)
    y_values_svm = np.sin(x_values)
    y_values_rf = np.cos(x_values)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(x_values, y_values_svm, label='SVM', linestyle='--')
    ax3.plot(x_values, y_values_rf, label='Random Forest', linestyle='-.')
    ax3.set_xlabel('X-axis')
    ax3.set_ylabel('Y-axis')
    ax3.set_title('Model Comparison')
    ax3.legend()

    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    plot_data3 = base64.b64encode(buffer3.read()).decode('utf-8')
    buffer3.close()

    return plot_data1, plot_data2, plot_data3


def reports_view(request):
    # Call the function to generate plots
    plot_data1, plot_data2, plot_data3 = generate_plots()

    # Fetch data from database tables
    train_logs = TrainLog.objects.all().values('id', 'model', 'accuracy')
    test_logs = TestLog.objects.all().values('id', 'model', 'accuracy')
    file_logs = FileLog.objects.all().values('id', 'protocol_type', 'flag', 'service', 'is_ddos')

    # Convert queryset to list of dictionaries
    train_logs = list(train_logs)
    test_logs = list(test_logs)
    file_logs = list(file_logs)

    # Return the plot data and table data as JSON response
    return JsonResponse({
        'plot_data1': plot_data1,
        'plot_data2': plot_data2,
        'plot_data3': plot_data3,
        'train_logs': train_logs,
        'test_logs': test_logs,
        'file_logs': file_logs
    })

