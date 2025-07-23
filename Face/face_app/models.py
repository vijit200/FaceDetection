from mongoengine import Document, StringField, DateField

class UserProfile(Document):
    image = StringField(required=True)  # base64 string
    username = StringField(required=True, max_length=100)
    email = StringField(required=True)
    address = StringField()
    phone = StringField()
    birth_date = DateField()

    def __str__(self):
        return f"{self.username} - {self.email}"
