// lib/screens/socket_service.dart
import 'package:socket_io_client/socket_io_client.dart' as IO;

class SocketService {
  late IO.Socket socket;
  final Function(Map<String, dynamic>) onStatusUpdate;
  final Function() onConnect; // <--- NEW: Callback for connection success

  SocketService({
    required this.onStatusUpdate,
    required this.onConnect, // <--- Add this
  });

  void initConnection() {
    // ⚠️ DOUBLE CHECK YOUR IP ADDRESS ⚠️
    const String serverUrl = 'http://192.168.1.4:5000';

    socket = IO.io(serverUrl, <String, dynamic>{
      'transports': ['websocket'],
      'autoConnect': false,
    });

    // 1. Listen for Connection Success
    socket.onConnect((_) {
      print('✅ Connected to Geist Backend');
      onConnect(); // <--- Trigger the UI update immediately
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
