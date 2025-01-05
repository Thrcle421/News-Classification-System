from django.core.management.base import BaseCommand
from News_prediction.code.AGNews_project import train_initial_model


class Command(BaseCommand):
    help = 'Trains the initial model'

    def handle(self, *args, **kwargs):
        self.stdout.write("Starting initial model training...")
        train_initial_model()
        self.stdout.write(self.style.SUCCESS(
            "Initial model training completed"))
