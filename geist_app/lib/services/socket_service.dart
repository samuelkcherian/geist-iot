// lib/services/socket_service.dart
import 'package:socket_io_client/socket_io_client.dart' as IO;
import 'package:shared_preferences/shared_preferences.dart';

class SocketService {
  late IO.Socket socket;
  final Function(Map<String, dynamic>) onStatusUpdate;
  final Function() onConnect; // <--- NEW: Callback for connection success

  SocketService({
    required this.onStatusUpdate,
    required this.onConnect, // <--- Add this
  });

  Future<void> initConnection() async {
    final prefs = await SharedPreferences.getInstance();
    String savedIp = prefs.getString('backend_ip') ?? '10.172.83.136';
    String serverUrl = 'http://$savedIp:5000';
    print('Connecting to Socket.IO at $serverUrl');

    socket = IO.io(serverUrl, <String, dynamic>{
      'autoConnect': false,
    });

    // 1. Listen for Connection Success
    socket.onConnect((_) {
      print('✅ Connected to Geist Backend');
      onConnect(); // <--- Trigger the UI update immediately
    });

    socket.onConnectError((data) {
      print('❌ Socket Connection Error: $data');
    });

    socket.onError((data) {
      print('❌ Socket Error: $data');
    });

    socket.onDisconnect((_) {
      print('🔌 Socket Disconnected');
    });

    // 2. Listen for Data
    socket.on('status_update', (data) {
      if (data != null) {
        onStatusUpdate(Map<String, dynamic>.from(data));
      }
    });

    socket.connect();
  }
}
