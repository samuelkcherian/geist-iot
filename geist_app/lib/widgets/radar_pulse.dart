// lib/widgets/radar_pulse.dart
import 'package:flutter/material.dart';

class RadarPulse extends StatefulWidget {
  final bool isAlarm;
  const RadarPulse({super.key, required this.isAlarm});

  @override
  State<RadarPulse> createState() => _RadarPulseState();
}

class _RadarPulseState extends State<RadarPulse>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      lowerBound: 0.5,
      duration: const Duration(seconds: 2),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Color pulseColor = widget.isAlarm ? Colors.red : Colors.greenAccent;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Stack(
          alignment: Alignment.center,
          children: [
            // Outer Ripple
            Container(
              width: 60 * _controller.value,
              height: 60 * _controller.value,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: pulseColor.withOpacity(1 - _controller.value),
              ),
            ),
            // Inner Circle
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: pulseColor,
                boxShadow: [
                  BoxShadow(color: pulseColor.withOpacity(0.6), blurRadius: 10),
                ],
              ),
              child: Icon(
                widget.isAlarm ? Icons.warning_rounded : Icons.radar,
                color: Colors.white,
                size: 20,
              ),
            ),
          ],
        );
      },
    );
  }
}
