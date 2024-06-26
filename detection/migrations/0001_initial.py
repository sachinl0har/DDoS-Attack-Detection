# Generated by Django 5.0.4 on 2024-04-28 16:44

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FileLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('protocol_type', models.CharField(max_length=500)),
                ('flag', models.CharField(max_length=500)),
                ('service', models.CharField(max_length=500)),
                ('is_ddos', models.CharField(max_length=500)),
            ],
        ),
        migrations.CreateModel(
            name='TestLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.CharField(max_length=500)),
                ('accuracy', models.CharField(max_length=500)),
            ],
        ),
        migrations.CreateModel(
            name='TrainLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.CharField(max_length=500)),
                ('accuracy', models.CharField(max_length=500)),
            ],
        ),
    ]
