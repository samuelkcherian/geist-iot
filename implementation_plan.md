# Phase 2: CNN Integration & UI Polish Plan

## Goal Description
Integrate the 4 CNN-classified states (`fall`, `walk`, `sit`, `empty`) into the Geist backend and frontend. The team is gathering hardware CSI data to train this CNN. Once complete, the CNN will output these 4 states to the backend via MQTT. 
Simultaneously, we will polish the Flutter UI to make the [ProfileScreen](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/profile_screen.dart#5-152) fully interactive like a real-world app and update the [HomeScreen](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/home_screen.dart#8-14) to beautifully visualize the 4 different states.

## Proposed Changes

### 1. Backend Updates
- **[MODIFY] [geist_backend/server.py](file:///d:/Internship%20projects/GEIST/geist/geist_backend/server.py)**
  - Update [on_mqtt_message](file:///d:/Internship%20projects/GEIST/geist/geist_backend/server.py#47-62) to parse the 4 new JSON states: `{"state": "walk"}`, `{"state": "sit"}`, `{"state": "empty"}`, `{"state": "fall"}`.
  - Emit corresponding `status_update` Socket.IO events with specific labels and colors (e.g., `Walk (Yellow)`, `Sit (Blue)`, `Empty (Green)`).

### 2. Frontend Real-time UI
- **[MODIFY] [geist_app/lib/screens/home_screen.dart](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/home_screen.dart)**
  - Update the [_updateUI](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/home_screen.dart#52-66) function to handle the 4 states smoothly.
  - Change the main gradient, pulse animation color, and descriptive text dynamically based on the state. For example:
    - **Walk**: "Subject Walking" (Yellow theme)
    - **Sit**: "Subject Seated" (Blue theme)
    - **Empty**: "Room Empty" (Green theme)
    - **Fall**: "CRITICAL FALL DETECTED" (Red theme)
  
### 3. Frontend Profile Interactivity
- **[MODIFY] [geist_app/lib/screens/profile_screen.dart](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/profile_screen.dart)**
  - Convert static list tiles into interactive items using `InkWell` or `ListTile`.
  - Create interactive bottom sheets or dialog modals for:
    - **Edit Profile**: A bottom sheet with text fields to rename the admin.
    - **Notifications**: A dialog to toggle push notifications.
    - **Log Out**: A confirmation dialog to gracefully return to the login screen and clear the saved IP.

## Verification Plan

### Manual Verification
1. Run [test_publish.py](file:///d:/Internship%20projects/GEIST/geist/geist_backend/test_publish.py) to rapidly cycle through `walk`, `sit`, `empty`, and `fall` states via MQTT.
2. Verify the [HomeScreen](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/home_screen.dart#8-14) updates cleanly and beautifully without visual jitter.
3. Navigate to the [ProfileScreen](file:///d:/Internship%20projects/GEIST/geist/geist_app/lib/screens/profile_screen.dart#5-152) and verify all buttons open their respective interactive modals.
