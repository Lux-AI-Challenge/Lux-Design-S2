import { showNotification } from '@mantine/notifications';

export function notifyError(title: string, message: string): void {
  showNotification({
    title,
    message,
    color: 'red',
  });
}
