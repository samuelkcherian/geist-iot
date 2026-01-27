import 'dart:typed_data'; // Needed for Int64List
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class NotificationService {
  final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  Future<void> init() async {
    // Android Initialization
    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const InitializationSettings initializationSettings =
        InitializationSettings(android: initializationSettingsAndroid);

    await _notificationsPlugin.initialize(initializationSettings);

    // --- DEFINE VIBRATION PATTERN ---
    // (Wait 0ms, Vibrate 1000ms, Wait 500ms, Vibrate 1000ms)
    final Int64List vibrationPattern = Int64List.fromList([0, 1000, 500, 1000]);

    // --- CREATE CHANNEL ---
    // ⚠️ NAME CHANGED to 'channel_v2' to force update on your phone!
    final AndroidNotificationChannel channel = AndroidNotificationChannel(
      'geist_emergency_v2', // <--- NEW ID
      'Critical Alerts', // <--- NEW NAME
      description: 'Vibrating alerts for detected falls',
      importance: Importance.max, // MAX means "Make noise and peek"
      playSound: true,
      enableVibration: true,
      vibrationPattern: vibrationPattern, // <--- Apply Pattern
    );

    await _notificationsPlugin
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.createNotificationChannel(channel);
  }

  Future<void> showFallAlert() async {
    // Make sure this matches the pattern above
    final Int64List vibrationPattern = Int64List.fromList([0, 1000, 500, 1000]);

    final AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
          'geist_emergency_v2', // <--- MUST MATCH NEW ID ABOVE
          'Critical Alerts',
          channelDescription: 'Critical Fall Detected',
          importance: Importance.max,
          priority: Priority.high,
          playSound: true,
          enableVibration: true,
          vibrationPattern:
              vibrationPattern, // <--- Apply Pattern to notification too
        );

    final NotificationDetails platformChannelSpecifics = NotificationDetails(
      android: androidPlatformChannelSpecifics,
    );

    await _notificationsPlugin.show(
      0,
      '⚠️ FALL DETECTED!',
      'Emergency: A fall has been detected in the living room.',
      platformChannelSpecifics,
    );
  }
}
