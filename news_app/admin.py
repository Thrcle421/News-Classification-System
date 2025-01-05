from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import NewsCategory, TrainedData, UntrainedData, UserProfile


@admin.register(NewsCategory)
class NewsCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    search_fields = ('name',)


@admin.register(TrainedData)
class TrainedDataAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'created_at', 'updated_at')
    list_filter = ('category', 'created_at')
    search_fields = ('title', 'content')
    date_hierarchy = 'created_at'


@admin.register(UntrainedData)
class UntrainedDataAdmin(admin.ModelAdmin):
    list_display = ('title', 'predicted_category', 'submitted_by',
                    'confidence_score', 'created_at', 'is_verified')
    list_filter = ('predicted_category', 'is_verified', 'created_at')
    search_fields = ('title', 'content')
    date_hierarchy = 'created_at'


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction_count', 'last_prediction')
    search_fields = ('user__username',)


# Unregister default User admin
admin.site.unregister(User)

# Create custom User admin


class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name',
                    'last_name', 'is_staff', 'is_active')
    list_filter = ('is_staff', 'is_active')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal Info', {'fields': ('first_name', 'last_name', 'email')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
        }),
        ('Important Dates', {'fields': ('last_login', 'date_joined')}),
    )


# Re-register User admin
admin.site.register(User, CustomUserAdmin)
