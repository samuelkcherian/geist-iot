// lib/screens/logs_screen.dart
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert'; // For converting JSON
import 'package:shared_preferences/shared_preferences.dart';
import '../utils/colors.dart';

class LogsScreen extends StatefulWidget {
  const LogsScreen({super.key});

  @override
  State<LogsScreen> createState() => _LogsScreenState();
}

class _LogsScreenState extends State<LogsScreen> {
  List<dynamic> _logs = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchLogs();
  }

  Future<void> _fetchLogs() async {
    final prefs = await SharedPreferences.getInstance();
    String savedIp = prefs.getString('backend_ip') ?? '192.168.1.4';
    String serverUrl = 'http://$savedIp:5000/api/logs';

    try {
      final response = await http.get(Uri.parse(serverUrl));

      if (response.statusCode == 200) {
        setState(() {
          _logs = json.decode(response.body);
          _isLoading = false;
        });
      }
    } catch (e) {
      print("Error fetching logs: $e");
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text("Event History"),
        backgroundColor: Colors.white,
        elevation: 0,
        titleTextStyle: const TextStyle(
          color: AppColors.darkText,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
        iconTheme: const IconThemeData(color: AppColors.darkText),
        actions: [
          IconButton(onPressed: _fetchLogs, icon: const Icon(Icons.refresh)),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _logs.isEmpty
          ? const Center(child: Text("No events recorded yet."))
          : ListView.builder(
              padding: const EdgeInsets.all(20),
              itemCount: _logs.length,
              itemBuilder: (context, index) {
                final log = _logs[index];

                // Split the string "Jan 25, 10:30 AM" into two parts
                // You might need to adjust this depending on your python format
                // For now, let's just display the full string smartly.

                return Container(
                  margin: const EdgeInsets.only(bottom: 15),
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(15),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.05),
                        blurRadius: 10,
                        offset: const Offset(0, 5),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          // Change color based on event type
                          color: log['event'].toString().contains("Fall")
                              ? Colors.redAccent.withOpacity(0.1)
                              : Colors.blueAccent.withOpacity(0.1),
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          log['event'].toString().contains("Fall")
                              ? Icons.warning
                              : Icons.info,
                          color: log['event'].toString().contains("Fall")
                              ? Colors.redAccent
                              : Colors.blueAccent,
                        ),
                      ),
                      const SizedBox(width: 15),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            log['event'],
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 5),
                          Row(
                            children: [
                              const Icon(
                                Icons.access_time,
                                size: 14,
                                color: Colors.grey,
                              ),
                              const SizedBox(width: 5),
                              Text(
                                log['time'], // This now contains "Jan 25, 10:30 AM"
                                style: const TextStyle(
                                  color: Colors.grey,
                                  fontSize: 13,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ],
                  ),
                );
              },
            ),
    );
  }
}
