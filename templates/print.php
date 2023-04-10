<?php
if($_POST['Submit']){
  $message = $_POST['Print_This'];
}
?>
<body onload="javascript: window.print();">
  <?php
  echo($message);
  ?>
</body>
